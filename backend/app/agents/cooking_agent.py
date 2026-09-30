import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from app.rag.retriever import search_substitutions_verified
from app.recipes.loader import load_recipe as parse_recipe
import asyncio
import time
import uuid

SIMPLE_PATTERNS = [
    "next", "done", "ok", "okay", "yes", "yeah", "yep", "sure",
    "got it", "ready", "finished", "continue", "go on", "next step",
    "what's next", "whats next", "repeat", "again", "say that again",
]


def is_simple_turn(text: str) -> bool:
    cleaned = text.lower().strip().rstrip(".!?")
    if len(cleaned.split()) > 4:
        return False
    return any(cleaned == p or cleaned.startswith(p + " ") for p in SIMPLE_PATTERNS)

SYSTEM_PROMPT = """You are Sous Chef, a hands-free voice cooking copilot. \
The user is cooking and cannot type or read — everything is spoken.

Personality:
- Warm, calm, encouraging. A patient friend who is a great cook.
- Keep responses SHORT. One or two sentences. The user has their hands full.
- Natural spoken language only.

How recipes work:
- When the user names a dish ("make chicken biryani", "let's do pasta"), call \
load_recipe with that dish name. The system will find a real recipe from the web \
automatically. You do NOT need a URL and you must NEVER ask the user for one.
- When the user wants you to choose ("you pick", "suggest something", "what should \
I make", or they name only a category like "something Italian"), call \
suggest_recipe first. Propose one concrete dish, confirm, THEN call load_recipe.
- If load_recipe fails after searching, say you couldn't find that one and suggest \
a different, more common dish. Never ask for a URL. Never invent a recipe.

Once a recipe is loaded:
- Briefly say what you found ("Found a chicken biryani, serves four") and offer to \
start. Don't read the whole ingredient list unless asked.
- Walk through ONE step at a time. Wait for the user to say next, done, or ready.
- Set timers when a step involves waiting. Use set_timer.
- Handle substitutions with search_substitution when they're missing something.
- Coordinate multiple dishes with schedule_dishes.
- The user can move around: next, back, repeat, or jump to a step number. \
Use the matching tool rather than reciting from memory.

What you never do:
- Never ask for a URL.
- Never invent a recipe or its steps. If the tools can't find it, say so.
- Never give long explanations unless asked.
- Never repeat yourself.

What you do:
- Save dietary constraints the instant they come up. If the user says they're \
vegan, gluten free, allergic to nuts, or anything similar, call set_constraint \
immediately, even if they mention it in passing while discussing something else. \
Acknowledge in a few words and carry on.
- Constraints persist for the whole session and are applied to every recipe \
search automatically. Do not re-ask.
- If a recipe can't be found under the active constraints, say so plainly and \
offer to adapt a standard recipe instead. Never quietly serve something that \
violates a saved constraint.

Speaking style (you are heard, not read):
- No bullet points, markdown, symbols, or abbreviations.
- Use contractions.
- Say numbers naturally: "three hundred fifty degrees", not "350F".
- Spell out fractions: "half a cup", not "1/2 cup".
- When asked about quantities or what ingredients are needed, call get_ingredients. \
Never guess amounts from memory when a recipe is loaded."""


TOOLS = [
    {
        "name": "load_recipe",
        "description": (
            "Load a recipe. Accepts a URL to a recipe website, or raw recipe text "
            "that the user pastes directly. Extracts ingredients and step-by-step "
            "instructions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "recipe_query": {
                    "type": "string",
                    "description": (
                        "The URL of a recipe page, or the raw recipe text the user "
                        "provided. Pass exactly what the user gave you."
                    ),
                }
            },
            "required": ["recipe_query"],
        },
    },
    {
        "name": "suggest_recipe",
        "description": (
            "Use when the user wants YOU to choose a recipe for them — 'you pick', "
            "'suggest something', 'what should I make', 'surprise me', or when they "
            "name a category but not a dish ('something Italian', 'a quick dinner'). "
            "Returns guidance to propose one concrete dish and confirm before loading."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cuisine_or_type": {
                    "type": "string",
                    "description": (
                        "Any constraint the user gave: cuisine, meal type, dietary "
                        "need, time limit. Empty string if they gave none."
                    ),
                }
            },
            "required": [],
        },
    },
        {
        "name": "get_ingredients",
        "description": (
            "Get the full ingredient list with quantities for the loaded recipe. "
            "Use whenever the user asks about amounts, quantities, what they need, "
            "or how much of something to use."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_current_step",
        "description": "Get the current step the user is on in the active recipe.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "set_constraint",
        "description": (
            "Save a dietary restriction or preference that should apply to every "
            "recipe from now on. Call this the moment the user mentions one — "
            "'I'm vegan', 'no gluten', 'I don't eat pork', 'dairy free', "
            "'nut allergy'. Call it even mid-conversation about something else."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "constraint": {
                    "type": "string",
                    "description": (
                        "Short phrase in the user's own terms: 'vegan', "
                        "'gluten free', 'no pork', 'nut allergy'."
                    ),
                }
            },
            "required": ["constraint"],
        },
    },
    {
        "name": "clear_constraint",
        "description": (
            "Remove a saved dietary constraint. Use when the user says something "
            "like 'actually I can eat dairy today' or 'forget the vegan thing'. "
            "Omit the constraint to clear all of them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "constraint": {
                    "type": "string",
                    "description": "Which one to remove. Empty removes all.",
                }
            },
        },
    },
    {
        "name": "list_constraints",
        "description": "Check what dietary constraints are currently saved.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "next_step",
        "description": "Move to the next step in the recipe.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
        {
        "name": "previous_step",
        "description": "Go back one step. Use for 'go back', 'previous step', 'wait, what was the last one'.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "repeat_step",
        "description": "Say the current step again without moving. Use for 'repeat that', 'say again', 'what was that', 'I missed it'.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "go_to_step",
        "description": "Jump to a specific step number. Use for 'go to step four', 'take me back to step two'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "step_number": {
                    "type": "integer",
                    "description": "The step number to jump to, counting from 1.",
                }
            },
            "required": ["step_number"],
        },
    },
    {
        "name": "set_timer",
        "description": "Set a countdown timer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {"type": "integer", "description": "Duration in seconds"},
                "label": {"type": "string", "description": "What the timer is for"},
            },
            "required": ["seconds", "label"],
        },
    },
    {
        "name": "search_substitution",
        "description": "Find a substitute for an ingredient the user doesn't have.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ingredient": {
                    "type": "string",
                    "description": "The ingredient to find a substitute for",
                }
            },
            "required": ["ingredient"],
        },
    },
        {
        "name": "cancel_timer",
        "description": "Cancel a running timer. Omit label to cancel all of them.",
        "input_schema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Which timer to cancel. Empty cancels all.",
                }
            },
        },
    },
    {
        "name": "list_timers",
        "description": "Check what timers are running and how much time is left.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "schedule_dishes",
        "description": "Calculate start times so multiple dishes finish together.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dishes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "total_minutes": {"type": "integer"},
                        },
                    },
                    "description": "List of dishes with their total cook times",
                }
            },
            "required": ["dishes"],
        },
    },
]



class CookingState(TypedDict):
    messages: Annotated[list, add_messages]


class CookingSession:
    def __init__(self, restored: dict | None = None):
        if restored:
            self.current_recipe = restored.get("current_recipe")
            self.current_step = restored.get("current_step", 0)
            self.active_timers = restored.get("active_timers", [])
            self.constraints = restored.get("constraints", [])
        else:
            self.current_recipe = None
            self.current_step = 0
            self.active_timers = []
            self.constraints = []

        self._alert_callback = None
        self._timer_tasks = []

    
    def _set_constraint(self, constraint: str) -> str:
        c = constraint.strip().lower()
        if not c:
            return json.dumps({"error": "No constraint given."})
        if c in self.constraints:
            return json.dumps({
                "already_set": True,
                "constraints": self.constraints,
            })
        self.constraints.append(c)
        return json.dumps({
            "saved": True,
            "constraints": self.constraints,
            "note": (
                "This now applies to every recipe search for the rest of the "
                "session. Acknowledge it briefly and move on."
            ),
        })

    def _clear_constraint(self, constraint: str = "") -> str:
        if not self.constraints:
            return json.dumps({"message": "No constraints were set."})
        if constraint:
            c = constraint.strip().lower()
            before = len(self.constraints)
            self.constraints = [
                x for x in self.constraints if c not in x and x not in c
            ]
            removed = before - len(self.constraints)
            return json.dumps({
                "removed": removed,
                "constraints": self.constraints,
            })
        self.constraints = []
        return json.dumps({"cleared_all": True, "constraints": []})

    def _list_constraints(self) -> str:
        return json.dumps({"constraints": self.constraints})
    def set_alert_callback(self, callback) -> None:
        """Called by the pipeline once it knows how to reach the browser."""
        self._alert_callback = callback

    def resume_timers(self) -> None:
        """After a reconnect, drop expired timers and restart the live ones."""
        now = time.time()
        still_running = []
        for t in self.active_timers:
            remaining = t.get("expires_at", 0) - now
            if remaining > 1:
                t["seconds"] = int(remaining)
                still_running.append(t)
                self._spawn_timer(t)
        self.active_timers = still_running

    def cancel_all_timers(self) -> None:
        for task in self._timer_tasks:
            task.cancel()
        self._timer_tasks = []

    def _suggest_recipe(self, cuisine_or_type: str) -> str:
        return json.dumps({
            "action": "suggest",
            "note": (
                "Suggest ONE specific, well-known dish that fits what the user wants "
                f"(context: '{cuisine_or_type}'). Say the dish name and one sentence "
                "on why it's a good pick. Then ask if they want to make it. Do NOT "
                "call load_recipe until they confirm. Pick something concrete and "
                "common so it's easy to find — for example 'spaghetti carbonara' or "
                "'chicken stir fry', not an obscure dish."
            ),
        })
    def _previous_step(self) -> str:
        if not self.current_recipe:
            return json.dumps({"error": "No recipe loaded yet."})
        if self.current_step <= 0:
            return json.dumps({
                "at_start": True,
                "message": "Already on the first step.",
                "instruction": self.current_recipe["steps"][0],
            })
        self.current_step -= 1
        return json.dumps({
            "step_number": self.current_step + 1,
            "total_steps": len(self.current_recipe["steps"]),
            "instruction": self.current_recipe["steps"][self.current_step],
        })

    def _repeat_step(self) -> str:
        return self._get_current_step()

    def _go_to_step(self, step_number: int) -> str:
        if not self.current_recipe:
            return json.dumps({"error": "No recipe loaded yet."})
        total = len(self.current_recipe["steps"])
        if step_number < 1 or step_number > total:
            return json.dumps({
                "error": f"Step {step_number} doesn't exist. This recipe has {total} steps.",
            })
        self.current_step = step_number - 1
        return json.dumps({
            "step_number": step_number,
            "total_steps": total,
            "instruction": self.current_recipe["steps"][self.current_step],
        })

    async def execute_tool(self, tool_name: str, tool_input: dict) -> str:
        try:
            if tool_name == "load_recipe":
                query = tool_input.get("recipe_query") or tool_input.get("recipe_name") or ""
                return await self._load_recipe(query)
            elif tool_name == "suggest_recipe":
                return self._suggest_recipe(tool_input.get("cuisine_or_type", ""))
            elif tool_name == "get_current_step":
                return self._get_current_step()
            elif tool_name == "next_step":
                return self._next_step()
            elif tool_name == "previous_step":
                return self._previous_step()
            elif tool_name == "set_constraint":
                return self._set_constraint(tool_input.get("constraint", ""))
            elif tool_name == "clear_constraint":
                return self._clear_constraint(tool_input.get("constraint", ""))
            elif tool_name == "list_constraints":
                return self._list_constraints()
            elif tool_name == "repeat_step":
                return self._repeat_step()
            elif tool_name == "go_to_step":
                return self._go_to_step(int(tool_input.get("step_number", 1)))
            elif tool_name == "set_timer":
                return self._set_timer(
                    tool_input.get("seconds", 0),
                    tool_input.get("label", "timer"),
                )
            elif tool_name == "cancel_timer":
                return self._cancel_timer(tool_input.get("label", ""))
            elif tool_name == "list_timers":
                return self._list_timers()
            elif tool_name == "search_substitution":
                return self._search_substitution(tool_input.get("ingredient", ""))
            elif tool_name == "schedule_dishes":
                return self._schedule_dishes(tool_input.get("dishes", []))
            elif tool_name == "get_ingredients":
                return self._get_ingredients()
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as e:
            print(f"Tool '{tool_name}' failed: {e}")
            return json.dumps({
                "error": f"That didn't work: {str(e)}",
                "recovery": "Tell the user something went wrong and ask them to try rephrasing.",
            })
        
        

    async def _load_recipe(self, query: str) -> str:
        recipe = await parse_recipe(query, constraints=self.constraints)

        if not recipe:
            msg = (
                "Couldn't find a recipe for that. If it was a URL, the site may "
                "be blocked or the page is gone. Suggest a different dish, or "
                "ask the user to paste the recipe text."
            )
            if self.constraints:
                msg += (
                    f" Note the active constraints ({', '.join(self.constraints)}) "
                    "narrowed the search — say so, and offer to adapt a standard "
                    "recipe instead."
                )
            return json.dumps({"found": False, "message": msg})

        self.current_recipe = {
            "name": recipe.name,
            "ingredients": recipe.ingredients,
            "steps": recipe.steps,
            "servings": recipe.servings,
            "total_minutes": recipe.total_minutes,
            "source": recipe.source,
        }
        self.current_step = 0

        return json.dumps({
            "found": True,
            "name": recipe.name,
            "servings": recipe.servings,
            "total_minutes": recipe.total_minutes,
            "total_steps": len(recipe.steps),
            "ingredients_count": len(recipe.ingredients),
            "first_step": recipe.steps[0] if recipe.steps else None,
            "parser_used": recipe.parser_used,
            "active_constraints": self.constraints,
        })

    def _get_current_step(self) -> str:
        if not self.current_recipe:
            return json.dumps({"error": "No recipe loaded yet."})
        step = self.current_recipe["steps"][self.current_step]
        return json.dumps({
            "step_number": self.current_step + 1,
            "total_steps": len(self.current_recipe["steps"]),
            "instruction": step,
        })

    def _next_step(self) -> str:
        if not self.current_recipe:
            return json.dumps({"error": "No recipe loaded yet."})
        self.current_step += 1
        if self.current_step >= len(self.current_recipe["steps"]):
            self.current_step = len(self.current_recipe["steps"]) - 1
            return json.dumps({
                "done": True,
                "message": "That was the last step. The dish is complete!",
            })
        step = self.current_recipe["steps"][self.current_step]
        return json.dumps({
            "step_number": self.current_step + 1,
            "total_steps": len(self.current_recipe["steps"]),
            "instruction": step,
        })

    def _set_timer(self, seconds: int, label: str) -> str:
        if seconds <= 0:
            return json.dumps({"error": "Timer duration must be positive."})

        timer = {
            "id": uuid.uuid4().hex[:8],
            "seconds": int(seconds),
            "label": label,
            "expires_at": time.time() + seconds,
        }
        self.active_timers.append(timer)
        self._spawn_timer(timer)

        mins, secs = divmod(int(seconds), 60)
        if mins and secs:
            spoken = f"{mins} minutes and {secs} seconds"
        elif mins:
            spoken = f"{mins} minutes"
        else:
            spoken = f"{secs} seconds"

        return json.dumps({
            "timer_set": True,
            "id": timer["id"],
            "duration": spoken,
            "label": label,
        })

    def _spawn_timer(self, timer: dict) -> None:
        try:
            task = asyncio.create_task(self._run_timer(timer))
            self._timer_tasks.append(task)
        except RuntimeError:
            print("No running event loop, timer will not fire")

    async def _run_timer(self, timer: dict) -> None:
        try:
            await asyncio.sleep(timer["seconds"])
        except asyncio.CancelledError:
            return

        self.active_timers = [
            t for t in self.active_timers if t["id"] != timer["id"]
        ]

        if self._alert_callback:
            try:
                await self._alert_callback(timer["label"])
            except Exception as e:
                print(f"Timer alert failed: {e}")

    def _cancel_timer(self, label: str = "") -> str:
        if not self.active_timers:
            return json.dumps({"cancelled": False, "message": "No active timers."})

        if label:
            match = next(
                (t for t in self.active_timers
                 if label.lower() in t["label"].lower()),
                None,
            )
            if not match:
                return json.dumps({
                    "cancelled": False,
                    "message": f"No timer matching '{label}'.",
                    "active": [t["label"] for t in self.active_timers],
                })
            self.active_timers.remove(match)
            return json.dumps({"cancelled": True, "label": match["label"]})

        self.cancel_all_timers()
        count = len(self.active_timers)
        self.active_timers = []
        return json.dumps({"cancelled": True, "count": count})

    def _list_timers(self) -> str:
        now = time.time()
        live = []
        for t in self.active_timers:
            remaining = int(t.get("expires_at", 0) - now)
            if remaining > 0:
                mins, secs = divmod(remaining, 60)
                live.append({
                    "label": t["label"],
                    "remaining": f"{mins} minutes {secs} seconds" if mins
                                 else f"{secs} seconds",
                })
        if not live:
            return json.dumps({"timers": [], "message": "No timers running."})
        return json.dumps({"timers": live})
    def _get_ingredients(self) -> str:
        if not self.current_recipe:
            return json.dumps({"error": "No recipe loaded yet."})
        return json.dumps({
            "recipe": self.current_recipe["name"],
            "servings": self.current_recipe.get("servings"),
            "ingredients": self.current_recipe["ingredients"],
        })

    def _search_substitution(self, ingredient: str) -> str:
        result = search_substitutions_verified(ingredient, top_k=3)
        verified = result["verified"]

        if not verified:
            return json.dumps({
                "found": False,
                "ingredient": ingredient,
                "message": (
                    f"The substitution database doesn't cover '{ingredient}'. "
                    "Ask what role it plays in the dish, then reason about alternatives "
                    "from first principles. Tell the user you're reasoning it out rather "
                    "than citing a known substitution."
                ),
            })

        return json.dumps({
            "found": True,
            "query": ingredient,
            "results": verified,
        })

    def _schedule_dishes(self, dishes: list) -> str:
        if not dishes:
            return json.dumps({"error": "No dishes provided."})
        max_time = max(d["total_minutes"] for d in dishes)
        schedule = []
        for d in dishes:
            offset = max_time - d["total_minutes"]
            if offset > 0:
                instruction = f"Start {d['name']} {offset} minutes after you begin"
            else:
                instruction = f"Start {d['name']} first, it takes the longest"
            schedule.append({
                "name": d["name"],
                "start_after_minutes": offset,
                "instruction": instruction,
            })
        schedule.sort(key=lambda x: x["start_after_minutes"])
        return json.dumps({
            "total_time_minutes": max_time,
            "schedule": schedule,
        })


def build_agent(session: CookingSession, model: str = "claude-sonnet-5"):
    llm = ChatAnthropic(
        model=model,
        max_tokens=300,
    )
    
    tools_for_llm = []
    for t in TOOLS:
        tools_for_llm.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        })

    llm_with_tools = llm.bind_tools(tools_for_llm)

    def call_model(state: CookingState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: CookingState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    async def call_tools(state: CookingState):
        last_message = state["messages"][-1]
        tool_results = []
        for tool_call in last_message.tool_calls:
            result = await session.execute_tool(
                tool_call["name"],
                tool_call["args"],
            )
            tool_results.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": tool_results}

    graph = StateGraph(CookingState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()