import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from app.rag.retriever import search_substitutions_verified
from app.recipes.loader import load_recipe as parse_recipe

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

What you never do:
- Never ask for a URL.
- Never invent a recipe or its steps. If the tools can't find it, say so.
- Never give long explanations unless asked.
- Never repeat yourself.

Speaking style (you are heard, not read):
- No bullet points, markdown, symbols, or abbreviations.
- Use contractions.
- Say numbers naturally: "three hundred fifty degrees", not "350F".
- Spell out fractions: "half a cup", not "1/2 cup"."""


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
        "name": "get_current_step",
        "description": "Get the current step the user is on in the active recipe.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
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
    def __init__(self):
        self.current_recipe = None
        self.current_step = 0
        self.active_timers = []

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
            elif tool_name == "set_timer":
                return self._set_timer(
                    tool_input.get("seconds", 0),
                    tool_input.get("label", "timer"),
                )
            elif tool_name == "search_substitution":
                return self._search_substitution(tool_input.get("ingredient", ""))
            elif tool_name == "schedule_dishes":
                return self._schedule_dishes(tool_input.get("dishes", []))
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as e:
            print(f"Tool '{tool_name}' failed: {e}")
            return json.dumps({
                "error": f"That didn't work: {str(e)}",
                "recovery": "Tell the user something went wrong and ask them to try rephrasing.",
            })

    async def _load_recipe(self, query: str) -> str:
        recipe = await parse_recipe(query)

        if not recipe:
            return json.dumps({
                "found": False,
                "message": (
                    "Couldn't parse a recipe from that. If it was a URL, the site "
                    "might not be supported or the page didn't load. Ask the user "
                    "to paste the recipe text directly, or try a different URL."
                ),
            })

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
        timer = {"seconds": seconds, "label": label}
        self.active_timers.append(timer)
        minutes = seconds // 60
        remaining_secs = seconds % 60
        if minutes > 0 and remaining_secs > 0:
            time_str = f"{minutes} minutes and {remaining_secs} seconds"
        elif minutes > 0:
            time_str = f"{minutes} minutes"
        else:
            time_str = f"{seconds} seconds"
        return json.dumps({
            "timer_set": True,
            "duration": time_str,
            "label": label,
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


def build_agent(session: CookingSession):
    llm = ChatAnthropic(
        model="claude-sonnet-5",
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