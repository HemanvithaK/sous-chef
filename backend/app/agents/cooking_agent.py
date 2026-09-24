import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from app.rag.retriever import search_substitutions_verified


SYSTEM_PROMPT = """You are Sous Chef, a voice-first cooking copilot.
You help home cooks through recipes hands-free while they cook.

Personality:
- Warm, calm, encouraging. A patient friend who is a great cook.
- Keep responses SHORT. 1-2 sentences max unless they ask for detail.
- Use natural cooking language.

What you do:
1. Walk through recipes step by step. One step at a time. Wait for the user to say next.
2. Set timers when a step involves waiting. Use the set_timer tool.
3. Handle substitutions when someone is missing an ingredient. Use search_substitution.
4. Coordinate timing for multiple dishes. Use schedule_dishes.
5. Answer general cooking questions.

What you DON'T do:
- Don't give long explanations unless asked.
- Don't list all ingredients upfront unless asked. Jump to step 1.
- Don't repeat yourself.

Voice considerations:
- Avoid bullet points, markdown, or special characters.
- Use contractions.
- Say numbers naturally: "three hundred fifty degrees" not "350F"."""


TOOLS = [
    {
        "name": "load_recipe",
        "description": "Load a recipe by name. Returns the full recipe with ingredients and steps.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recipe_name": {
                    "type": "string",
                    "description": "Name of the recipe to load",
                }
            },
            "required": ["recipe_name"],
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


SAMPLE_RECIPES = {
    "thai basil chicken": {
        "name": "Thai Basil Chicken (Pad Kra Pao)",
        "servings": 2,
        "ingredients": [
            "1 lb ground chicken",
            "4 cloves garlic, minced",
            "3-4 Thai chilies, minced",
            "1 cup Thai basil leaves",
            "2 tbsp oyster sauce",
            "1 tbsp soy sauce",
            "1 tbsp fish sauce",
            "1 tsp sugar",
            "2 tbsp vegetable oil",
        ],
        "steps": [
            "Mince the garlic and chilies. Pick the basil leaves off the stems. Mix the oyster sauce, soy sauce, fish sauce, and sugar in a small bowl.",
            "Heat your wok or large skillet over high heat until it's smoking. Add the oil.",
            "Add garlic and chilies. Stir-fry for about thirty seconds until fragrant. Don't let the garlic burn.",
            "Add the ground chicken. Break it up and cook until no longer pink, about three to four minutes. Keep the heat high for browning.",
            "Pour in the sauce mixture. Toss everything together for about one minute.",
            "Kill the heat. Toss in the basil leaves and stir until they just wilt. Serve over jasmine rice.",
        ],
    },
    "pasta aglio e olio": {
        "name": "Spaghetti Aglio e Olio",
        "servings": 2,
        "ingredients": [
            "8 oz spaghetti",
            "6 cloves garlic, thinly sliced",
            "1/3 cup extra virgin olive oil",
            "1/2 tsp red pepper flakes",
            "1/4 cup fresh parsley, chopped",
            "Salt to taste",
        ],
        "steps": [
            "Bring a large pot of well-salted water to a rolling boil. Add the spaghetti and cook one minute short of the package time.",
            "While the pasta cooks, heat the olive oil in a large skillet over medium-low heat. Add the sliced garlic and red pepper flakes. Cook slowly, stirring often, until the garlic is golden. About four to five minutes. Low and slow.",
            "Before you drain the pasta, scoop out about half a cup of the starchy pasta water. You'll need this.",
            "Add the drained pasta directly to the garlic oil. Add a splash of pasta water. Toss aggressively over medium heat for about a minute. The starch and oil should come together into a silky sauce.",
            "Kill the heat, toss in the parsley, and hit it with a final drizzle of good olive oil. Serve immediately.",
        ],
    },
    "scrambled eggs": {
        "name": "Perfect Scrambled Eggs",
        "servings": 1,
        "ingredients": [
            "3 large eggs",
            "1 tbsp butter",
            "Salt and pepper to taste",
            "1 tbsp chives, chopped (optional)",
        ],
        "steps": [
            "Crack three eggs into a bowl. Don't add salt yet. Whisk until the yolks and whites are fully combined.",
            "Put a non-stick pan on medium-low heat. Add the butter and let it melt without browning.",
            "Pour in the eggs. Wait about thirty seconds, then start pushing them gently with a spatula from the edges toward the center. Big, slow folds.",
            "When the eggs are about seventy percent set but still look wet, pull the pan off the heat. The residual heat finishes them. Season with salt and pepper now.",
            "Slide onto a plate, top with chives if you have them. Eat immediately. Scrambled eggs wait for no one.",
        ],
    },
}


class CookingState(TypedDict):
    messages: Annotated[list, add_messages]


class CookingSession:
    def __init__(self):
        self.current_recipe = None
        self.current_step = 0
        self.active_timers = []

    def execute_tool(self, tool_name: str, tool_input: dict) -> str:
        if tool_name == "load_recipe":
            return self._load_recipe(tool_input["recipe_name"])
        elif tool_name == "get_current_step":
            return self._get_current_step()
        elif tool_name == "next_step":
            return self._next_step()
        elif tool_name == "set_timer":
            return self._set_timer(tool_input["seconds"], tool_input["label"])
        elif tool_name == "search_substitution":
            return self._search_substitution(tool_input["ingredient"])
        elif tool_name == "schedule_dishes":
            return self._schedule_dishes(tool_input["dishes"])
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _load_recipe(self, recipe_name: str) -> str:
        key = recipe_name.lower().strip()
        for name, recipe in SAMPLE_RECIPES.items():
            if key in name or name in key:
                self.current_recipe = recipe
                self.current_step = 0
                return json.dumps({
                    "found": True,
                    "name": recipe["name"],
                    "servings": recipe["servings"],
                    "total_steps": len(recipe["steps"]),
                    "ingredients": recipe["ingredients"],
                    "first_step": recipe["steps"][0],
                })
        available = list(SAMPLE_RECIPES.keys())
        return json.dumps({
            "found": False,
            "message": f"No recipe found for '{recipe_name}'.",
            "available_recipes": available,
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

    def call_tools(state: CookingState):
        last_message = state["messages"][-1]
        tool_results = []
        for tool_call in last_message.tool_calls:
            result = session.execute_tool(
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