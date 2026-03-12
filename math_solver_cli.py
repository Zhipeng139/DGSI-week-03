import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from sympy import Eq, Symbol, latex, factor
from sympy.core.sympify import SympifyError
from sympy.solvers import solve
# Added transformations for better parsing of implicit multiplication (e.g. 2x -> 2*x)
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


load_dotenv()

console = Console()
client: OpenAI | None = None

MODEL = os.getenv("MODEL", "gpt-4.1-mini")
OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
You are a high-school math assistant.
Use tools when computation, equation solving, factoring, or plotting is needed.
Prefer concise, clear explanations suitable for students.
If no tool is required, answer directly.
"""

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_expression",
            "description": "Evaluate a math expression numerically or symbolically, optionally substituting variable values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Expression to evaluate, for example: 2*(3+4), sin(pi/6), x^2 + 3*x with substitutions.",
                    },
                    "substitutions": {
                        "type": "object",
                        "description": "Optional variable substitutions as key-value pairs, for example {\"x\": 2}.",
                        "additionalProperties": {"type": "number"},
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_equation",
            "description": "Solve an equation for a given variable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {
                        "type": "string",
                        "description": "Equation string, for example: 2*x + 3 = 11 or x^2 - 5*x + 6 = 0.",
                    },
                    "variable": {
                        "type": "string",
                        "description": "Variable to solve for, for example: x.",
                    },
                },
                "required": ["equation", "variable"],
            },
        },
    },
    {
        # Added factor_expression tool to cover factorization test cases
        "type": "function",
        "function": {
            "name": "factor_expression",
            "description": "Factor a mathematical polynomial expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Polynomial expression to factor, for example: x^2 + 7*x + 12.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_function",
            "description": "Plot a single-variable function and save the graph as a PNG image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Function expression in one variable, for example: x^2 - 4*x + 3.",
                    },
                    "variable": {
                        "type": "string",
                        "description": "Independent variable symbol, for example: x.",
                    },
                    "x_min": {
                        "type": "number",
                        "description": "Minimum x value for plotting.",
                    },
                    "x_max": {
                        "type": "number",
                        "description": "Maximum x value for plotting.",
                    },
                    "num_points": {
                        "type": "integer",
                        "description": "Number of sampled points. Larger values create smoother plots.",
                        "minimum": 50,
                        "maximum": 5000,
                        "default": 400,
                    },
                },
                "required": ["expression", "variable", "x_min", "x_max"],
            },
        },
    },
]

# Combined transformations to handle implicit multiplication safely
TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

def _parse_safely(expr_str: str):
    """Safely parse strings with implicit multiplications into SymPy objects."""
    normalized = expr_str.replace("^", "**")
    return parse_expr(normalized, transformations=TRANSFORMATIONS)


def evaluate_expression(expression: str, substitutions: dict | None = None) -> str:
    try:
        expr = _parse_safely(expression)
        if substitutions:
            subs_map = {Symbol(k): v for k, v in substitutions.items()}
            expr = expr.subs(subs_map)
        simplified = expr.simplify()
        numeric = simplified.evalf()
        payload = {
            "input_expression": expression,
            "substitutions": substitutions or {},
            "symbolic_result": str(simplified),
            "numeric_result": str(numeric),
        }
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"Evaluation failed: {exc}"}, ensure_ascii=False)


def solve_equation(equation: str, variable: str) -> str:
    try:
        if "=" not in equation:
            return json.dumps({"error": "Equation must contain '=' sign."}, ensure_ascii=False)
        left, right = equation.split("=", maxsplit=1)
        left_expr = _parse_safely(left.strip())
        right_expr = _parse_safely(right.strip())
        symbol = Symbol(variable)
        solutions = solve(Eq(left_expr, right_expr), symbol)
        payload = {
            "input_equation": equation,
            "variable": variable,
            "solutions": [str(sol) for sol in solutions],
            "solutions_latex": [latex(sol) for sol in solutions],
        }
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"Solving failed: {exc}"}, ensure_ascii=False)

def factor_expression(expression: str) -> str:
    """New tool specifically for factoring algebraic expressions."""
    try:
        expr = _parse_safely(expression)
        factored_expr = factor(expr)
        payload = {
            "input_expression": expression,
            "factored_result": str(factored_expr),
            "factored_latex": latex(factored_expr),
        }
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"Factoring failed: {exc}"}, ensure_ascii=False)


def plot_function(expression: str, variable: str, x_min: float, x_max: float, num_points: int = 400) -> str:
    try:
        if x_min >= x_max:
            return json.dumps({"error": "x_min must be smaller than x_max."}, ensure_ascii=False)
        if num_points < 50 or num_points > 5000:
            return json.dumps({"error": "num_points must be between 50 and 5000."}, ensure_ascii=False)

        var_symbol = Symbol(variable)
        expr = _parse_safely(expression)
        x_values = np.linspace(x_min, x_max, num_points)
        y_values = np.array([float(expr.subs(var_symbol, x).evalf()) for x in x_values], dtype=float)

        plots_dir = Path("plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"plot_{variable}_{abs(hash((expression, x_min, x_max, num_points))) % 10_000_000}.png"
        output_path = plots_dir / file_name

        plt.figure(figsize=(8, 5))
        plt.plot(x_values, y_values, label=f"y = {expression}")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlabel(variable)
        plt.ylabel("y")
        plt.title(f"f({variable}) = {expression}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        payload = {
            "expression": expression,
            "variable": variable,
            "x_range": [x_min, x_max],
            "num_points": num_points,
            "plot_file": str(output_path),
        }
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"Plotting failed: {exc}"}, ensure_ascii=False)


# Register the new tool in the map
TOOL_MAP = {
    "evaluate_expression": evaluate_expression,
    "solve_equation": solve_equation,
    "factor_expression": factor_expression,
    "plot_function": plot_function,
}


def create_message_panel(role: str, content: str) -> Panel:
    styles = {
        "user": ("bright_white on blue", "blue", "🧑 Student"),
        "assistant": ("bright_white on dark_green", "green", "🤖 Math Solver"),
        "system": ("bright_white on purple4", "magenta", "⚙️ System"),
        "tool": ("black on yellow", "yellow", "🔧 Tool Result"),
    }
    text_style, border_color, title = styles.get(role, ("bright_white on grey23", "white", role))
    return Panel(
        Text(content, style=text_style),
        title=title,
        title_align="left",
        border_style=border_color,
        padding=(0, 1),
    )


def show_context_stack(messages: list, tools_available: bool) -> Panel:
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold bright_white on grey23",
        style="on grey23",
    )
    table.add_column("#", style="bright_cyan on grey23", width=3)
    table.add_column("Role", style="bright_magenta on grey23", width=12)
    table.add_column("Content Preview", style="bright_white on grey23")

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        preview = content.replace("\n", " ") if content else "(tool_calls)"
        table.add_row(str(i), role, preview[:180])

    tool_names = ", ".join(tool["function"]["name"] for tool in AVAILABLE_TOOLS)
    tools_text = f"✅ Tools: [{tool_names}]" if tools_available else "❌ No Tools"

    return Panel(
        table,
        title=f"📚 Context Stack ({len(messages)} messages) | {tools_text}",
        border_style="magenta",
        style="on grey23",
        padding=(0, 1),
    )


def show_api_request(request_data: dict) -> Panel:
    json_str = json.dumps(request_data, indent=2, ensure_ascii=False)
    syntax = Syntax(json_str, "json", theme="monokai", background_color="grey23", word_wrap=True)
    return Panel(
        syntax,
        title="📤 API Request (sent to OpenAI)",
        border_style="yellow",
        style="on grey23",
        padding=(0, 1),
    )


def show_api_response(response_data: dict) -> Panel:
    json_str = json.dumps(response_data, indent=2, ensure_ascii=False)
    syntax = Syntax(json_str, "json", theme="monokai", background_color="grey23", word_wrap=True)
    return Panel(
        syntax,
        title="📥 API Response (from OpenAI)",
        border_style="cyan",
        style="on grey23",
        padding=(0, 1),
    )


def wait_for_llm() -> Live:
    return Live(
        Panel(
            Spinner("dots", text=Text(" Waiting for LLM response...", style="bold black on yellow")),
            border_style="yellow",
            style="on yellow",
            padding=(0, 1),
        ),
        console=console,
        refresh_per_second=10,
    )


def execute_tool_call(tool_call) -> str:
    function_name = tool_call.function.name
    console.print()
    console.print(Panel(Text(f"[TOOL CALL] {function_name}", style="bold black on yellow"), border_style="yellow", style="on yellow"))
    console.print(Panel(Text(f"[TOOL ARGS] {tool_call.function.arguments}", style="bold black on yellow"), border_style="yellow", style="on yellow"))
    try:
        function_args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as exc:
        raw_result = json.dumps({"error": f"Invalid tool arguments JSON: {exc}"}, ensure_ascii=False)
    else:
        function_impl = TOOL_MAP.get(function_name)
        if not function_impl:
            raw_result = json.dumps({"error": f"Unknown tool: {function_name}"}, ensure_ascii=False)
        else:
            try:
                raw_result = function_impl(**function_args)
            except Exception as exc:
                raw_result = json.dumps({"error": f"Tool execution failed: {exc}"}, ensure_ascii=False)
    console.print(Panel(Text(f"[TOOL RAW RESULT] {raw_result}", style="black on yellow"), border_style="yellow", style="on yellow"))
    return raw_result


def process_user_problem(messages: list) -> list:
    request_data = {"model": MODEL, "messages": messages, "tools": AVAILABLE_TOOLS, "temperature": 0}
    if OPENAI_API_ENDPOINT:
        request_data["_endpoint"] = OPENAI_API_ENDPOINT
    console.print()
    console.print(show_api_request(request_data))

    with wait_for_llm():
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=AVAILABLE_TOOLS, temperature=0)

    assistant_message = response.choices[0].message
    response_data = {
        "id": response.id,
        "model": response.model,
        "finish_reason": response.choices[0].finish_reason,
        "message": {"role": "assistant", "content": assistant_message.content, "tool_calls": None},
    }

    if assistant_message.tool_calls:
        response_data["message"]["tool_calls"] = [
            {"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in assistant_message.tool_calls
        ]

    console.print()
    console.print(show_api_response(response_data))

    if not assistant_message.tool_calls:
        messages.append({"role": "assistant", "content": assistant_message.content})
        console.print()
        console.print(create_message_panel("assistant", assistant_message.content or ""))
        return messages

    messages.append(
        {
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in assistant_message.tool_calls
            ],
        }
    )

    if assistant_message.content:
        console.print()
        console.print(create_message_panel("assistant", assistant_message.content))

    for tool_call in assistant_message.tool_calls:
        function_name = tool_call.function.name
        raw_result = execute_tool_call(tool_call)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": raw_result,
            }
        )
        console.print()
        console.print(create_message_panel("tool", raw_result))

    follow_request = {"model": MODEL, "messages": messages, "tools": AVAILABLE_TOOLS, "temperature": 0}
    if OPENAI_API_ENDPOINT:
        follow_request["_endpoint"] = OPENAI_API_ENDPOINT
    console.print()
    console.print(show_api_request(follow_request))

    with wait_for_llm():
        follow_up = client.chat.completions.create(model=MODEL, messages=messages, tools=AVAILABLE_TOOLS, temperature=0)

    follow_content = follow_up.choices[0].message.content or "I could not produce a final answer."
    messages.append({"role": "assistant", "content": follow_content})
    console.print()
    console.print(
        show_api_response(
            {
                "id": follow_up.id,
                "model": follow_up.model,
                "finish_reason": follow_up.choices[0].finish_reason,
                "message": {"role": "assistant", "content": follow_content},
            }
        )
    )
    console.print()
    console.print(create_message_panel("assistant", follow_content))
    return messages


def run_chat(single_problem: str | None = None) -> None:
    console.print()
    console.print(
        Panel(
            Text(
                "Scenario: High-School Math Solver\n\nAsk any expression, equation, or plotting problem.\nThe assistant can call math tools when needed.",
                style="bold bright_white on dark_green",
            ),
            title="🎓 Starting Math Session",
            border_style="green",
            style="on dark_green",
            padding=(1, 2),
        )
    )
    console.print()
    console.print(
        Panel(
            Syntax(json.dumps(AVAILABLE_TOOLS, indent=2), "json", theme="monokai", background_color="grey23", word_wrap=True),
            title="🔧 Tools Available to the Math Solver",
            border_style="cyan",
            style="on grey23",
            padding=(0, 1),
        )
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    console.print()
    console.print(show_context_stack(messages, tools_available=True))

    if single_problem:
        messages.append({"role": "user", "content": single_problem})
        console.print()
        console.print(create_message_panel("user", single_problem))
        messages = process_user_problem(messages)
        console.print()
        console.print(show_context_stack(messages, tools_available=True))
        return

    while True:
        console.print()
        user_input = console.input("[bold blue]🧑 Student: [/bold blue]")
        if not user_input.strip():
            console.print()
            console.print(
                Panel(
                    Text("Session ended. Returning to menu...", style="bold bright_white on grey23"),
                    border_style="dim",
                    style="on grey23",
                )
            )
            break
        messages.append({"role": "user", "content": user_input})
        console.print(create_message_panel("user", user_input))
        messages = process_user_problem(messages)
        console.print()
        console.print(show_context_stack(messages, tools_available=True))


def show_menu() -> None:
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                "[bold bright_white]🎓 OpenAI Function Calling Demo[/bold bright_white]\n\n"
                "[bright_cyan]High-School Math Solver Edition[/bright_cyan]\n\n"
                "[bright_white]Use rich CLI panels to inspect context,\nrequests, responses, and tool execution.[/bright_white]"
            ),
            title="🧮 Menu 🧮",
            title_align="center",
            border_style="bright_magenta",
            style="on grey23",
            padding=(1, 4),
        )
    )

    console.print()
    menu_table = Table(box=box.ROUNDED, show_header=False, style="on grey23", border_style="cyan")
    menu_table.add_column("Option", style="bold bright_cyan on grey23", width=5)
    menu_table.add_column("Description", style="bright_white on grey23")
    menu_table.add_row("1", "Interactive math chat")
    menu_table.add_row("2", "Single-shot math problem")
    menu_table.add_row("q", "Quit")
    console.print(Panel(menu_table, border_style="cyan", style="on grey23", padding=(0, 1)))


def build_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not found in .env")
    if OPENAI_API_ENDPOINT:
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_ENDPOINT)
    return OpenAI(api_key=OPENAI_API_KEY)


def main() -> None:
    global client

    parser = argparse.ArgumentParser(description="High-School Math Solver CLI")
    parser.add_argument("--problem", type=str, help="Single-shot math problem")
    args = parser.parse_args()

    console.clear()
    try:
        client = build_client()
    except Exception as exc:
        console.print(
            Panel(
                Text(f"Startup error:\n{exc}", style="bold bright_white on dark_red"),
                title="❌ Error",
                border_style="red",
                style="on dark_red",
            )
        )
        return

    console.print(
        Panel(
            Text(f"Model: {MODEL}\nEndpoint: {OPENAI_API_ENDPOINT or 'https://api.openai.com/v1'}", style="bright_white on grey23"),
            title="⚙️ Configuration",
            border_style="cyan",
            style="on grey23",
        )
    )

    if args.problem:
        run_chat(single_problem=args.problem)
        return

    while True:
        show_menu()
        console.print()
        choice = console.input("[bold cyan]Choose option (1/2/q): [/bold cyan]").strip().lower()
        if choice == "1":
            run_chat()
        elif choice == "2":
            console.print()
            one_problem = console.input("[bold blue]🧑 Enter your math problem: [/bold blue]").strip()
            if one_problem:
                run_chat(single_problem=one_problem)
        elif choice == "q":
            console.print()
            console.print(
                Panel(
                    Text("Goodbye! 🎓👋", style="bold bright_white on grey23"),
                    border_style="magenta",
                    style="on grey23",
                )
            )
            break
        else:
            console.print()
            console.print(
                Panel(
                    Text("Invalid option. Please choose 1, 2, or q.", style="bold black on yellow"),
                    border_style="yellow",
                    style="on yellow",
                )
            )


if __name__ == "__main__":
    main()