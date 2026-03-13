# Function Calling & LLM Agents Workshop Report

## Week 3: Math Solver Project

---

## 1) Three Little Pigs Demo Test

### 1.1 Configuration with `.env`

The demo uses environment variables loaded from a `.env` file through `python-dotenv`.  
The key fields are:

- `OPENAI_API_KEY` — authentication token for API access.
- `OPENAI_API_ENDPOINT` — optional OpenAI-compatible base URL.
- `MODEL` — model identifier selected for chat completion.

Typical setup:

```env
OPENAI_API_KEY=your-api-key-here
MODEL=gpt-4.1-mini
OPENAI_API_ENDPOINT=https://your-compatible-endpoint/v1
```

In both scripts, environment variables are loaded early and then used to initialize the OpenAI client. This keeps credentials outside source code and supports endpoint/model switching without changing Python logic.

### 1.2 Interactive Execution and the `rich` Terminal UI

The script uses the `rich` library to create an interactive terminal UI. The user plays the role of the wolf natively typing prompts, while the LLM acts as one of the three little pigs.

When executed, a menu offers two scenarios:
1. Scenario 1: Chat WITHOUT function calling (pig can only talk)
2. Scenario 2: Chat WITH function calling (pig can call hunter)

### 1.3 Output Differences: No-Tools vs Tools-Enabled

In the **no-tools** scenario (Scenario 1):

- The model can only generate text based on the conversation history.
- It may say something like “I am calling the hunter,” but no real function executes.
- No tool payload (`tool_calls`) is returned, keeping actions as mere narrative intent.

In the **tools-enabled** scenario (Scenario 2):

- The same conversational context includes a declared tool schema (`call_hunter`), which takes `urgency` (low/medium/high/emergency) and `message` as JSON arguments.
- When needed, the assistant response suspends text generation and outputs `tool_calls` with JSON arguments (e.g., `{"urgency": "emergency", "message": "The wolf is trying to blow my house down!"}`).
- The host script parses this payload, runs the python function `call_hunter()`, mapping the urgency to an actual outcome (e.g., "The hunter is sprinting to your location..."), and returns a concrete tool result.
- A second model call integrates this tool output into the final assistant message.

This demonstrates the practical difference between **narrative intent** and **executable action**.

### 1.4 Why the Hunter Was Called Only Under Threat

The model calls the hunter only when threat language appears because:

- The system prompt explicitly instructs the pig: *"IMPORTANT: If you have access to tools and you are in danger, USE THEM! Call the hunter immediately if the wolf threatens you!"*
- Threat cues such as “I am the wolf” and “I will blow your house” satisfy that condition.
- In non-threatening dialog, the policy does not justify escalation, so no tool is called.

In short, tool usage is not random; it is **policy-conditioned behavior** driven by prompt constraints and the user context.

---

## 2) Function Calling Explanation

### 2.1 Layman’s Explanation

Function calling means the model can realize:

> “I need to use this specific external tool (like calling the hunter with a specific urgency, or using a math solver) to properly respond to the user's situation.”

Instead of just inventing everything in plain text, it asks the host app to run a real function with structured, precise inputs.

### 2.2 Text-Only Response vs `tool_calls`

- **Text-only response**: The assistant returns normal natural language content only.
- **`tool_calls` response**: The assistant returns a payload requesting to run one or more tools, including:
  - The exact tool name (e.g., `call_hunter`)
  - The structured JSON arguments (e.g., `{"urgency": "high", "message": "Help!"}`)
  - A unique tool call ID

When a `tool_calls` payload exists, the host program effectively pauses the LLM, executes the local code, and sends the real-world results back for a final, informed answer.

### 2.3 Why the Host Program Stays in Control

The Python host remains the controller for safety and reliability:

- It alone decides which declared tools are actually available and executable.
- It validates/decodes tool arguments (e.g., ensuring `urgency` is one of the allowed enums).
- It handles exceptions and converts failures into safe outputs injected back into the context.
- It can reject unknown tools or malformed arguments.

Therefore, the model can **request** actions based on its reasoning, but it cannot independently execute arbitrary system operations.

---

## 3) Math Solver Design

### 3.1 Implemented Tool Set

The Week 3 solver uses four tools:

1. `evaluate_expression`
2. `solve_equation`
3. `factor_expression`
4. `plot_function`

### 3.2 Why a Small, Well-Defined Tool Set

The design intentionally keeps tools minimal:

- Reduces ambiguity in tool selection.
- Makes schema descriptions clearer.
- Simplifies validation and error handling.
- Improves reproducibility for workshop learners.

This follows first-principles engineering: only add capabilities required by the core user goals (evaluate, solve, plot).

### 3.3 Key Python Snippets

#### A) Tool schema declaration

```python
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "solve_equation",
            "description": "Solve an equation for a given variable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {"type": "string"},
                    "variable": {"type": "string"},
                },
                "required": ["equation", "variable"],
            },
        },
    }
]
```

#### B) Safe equation solving with SymPy

```python
def solve_equation(equation: str, variable: str) -> str:
    # Validate the expected equation format first.
    if "=" not in equation:
        return json.dumps({"error": "Equation must contain '=' sign."})

    left, right = equation.split("=", maxsplit=1)
    left_expr = sympify(left.replace("^", "**").strip())
    right_expr = sympify(right.replace("^", "**").strip())
    symbol = Symbol(variable)
    solutions = solve(Eq(left_expr, right_expr), symbol)

    # Return structured JSON for deterministic downstream handling.
    return json.dumps({"solutions": [str(s) for s in solutions]})
```

#### C) Host-side orchestration logic

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=AVAILABLE_TOOLS,
    temperature=0,
)

assistant_message = response.choices[0].message

# If no tool is needed, return direct answer.
if not assistant_message.tool_calls:
    return assistant_message.content

# Execute each requested tool call in the host program.
for tool_call in assistant_message.tool_calls:
    args = json.loads(tool_call.function.arguments)
    raw_result = TOOL_MAP[tool_call.function.name](**args)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.function.name,
        "content": raw_result,
    })
```

---

## 4) Testing Evidence

### 4.1 Test Execution Protocol

All end-to-end program tests were executed with the required command pattern:

```bash
uv run python <app>.py
```

For this project, the concrete command used was:

```bash
uv run python math_solver_cli.py --problem "Solve x^2 - 5*x + 6 = 0 and explain."
```

This validates the actual CLI application path (not only isolated function imports), including environment loading, model orchestration, tool selection, tool execution, and final response rendering.

### 4.2 Successful Cases

#### Arithmetic

CLI call:

```bash
uv run python math_solver_cli.py --problem "evaluate exactly: 2*(3+4). Return concise explanation."
```

Input problem:

- `2*(3+4)`

Observed tool output (`role="tool"`):

```json
{"input_expression":"2*(3+4)","substitutions":{},"symbolic_result":"14","numeric_result":"14.0000000000000"}
```

#### Algebra

CLI call:

```bash
uv run python math_solver_cli.py --problem "solve the equation x^2 - 5*x + 6 = 0."
```

Input equation:

- $x^2 - 5x + 6 = 0$

Observed tool output (`role="tool"`):

```json
{"input_equation":"x^2 - 5*x + 6 = 0","variable":"x","solutions":["2","3"],"solutions_latex":["2","3"]}
```

#### Plotting

CLI call:

```bash
uv run python math_solver_cli.py --problem "plot f(x)=x^2-4*x+3 for x from -2 to 6 and summarize."
```

Input function:

- $f(x)=x^2-4x+3,\;x\in[-2,6]$

Observed tool output (`role="tool"`):

```json
{"expression":"x^2 - 4*x + 3","variable":"x","x_range":[-2,6],"num_points":400,"plot_file":"plots/plot_x_5412279.png"}
```

Image placeholder:

- [INSERT_PLOT_IMAGE_HERE]

### 4.3 Failure Case and Graceful Handling

CLI call:

```bash
uv run python math_solver_cli.py --problem "evaluate this malformed expression: 2*/3. Explain the error briefly."
```

Failure input:

- `2*/3`

Observed tool behavior (`role="tool"`):

```json
{"error":"Evaluation failed: invalid syntax (<string>, line 1)"}
```

The program did not crash. It captured the parse/evaluation issue and returned a structured error payload, allowing the conversation flow to continue safely.

---

## 5) Required Questions (Explicit Technical Answers)

### Q1) Why is function calling more reliable than “just do the math” in plain text?

Because computation is delegated to deterministic math libraries (SymPy) instead of relying purely on generative token prediction. This reduces hallucinated intermediate steps and improves repeatability.

### Q2) Why should the tool set be small and well-defined?

A smaller tool surface lowers selection ambiguity, simplifies schema quality, and makes host-side validation easier. It also reduces accidental misuse and improves educational clarity.

### Q3) What is the role of `sympy`?

`sympy` is the symbolic mathematics engine used to parse expressions, simplify forms, and solve equations (including exact symbolic outputs where appropriate).

### Q4) What is the role of `matplotlib`?

`matplotlib` renders function graphs and exports them as `.png` files into the `plots/` directory, turning symbolic math into visual evidence.

### Q5) Describe the step-by-step lifecycle from user input to final answer.

1. User submits a natural-language math request.
2. Host sends messages + tool schemas to the model (first call).
3. Model either returns direct text or returns `tool_calls`.
4. Host validates/parses tool arguments and executes mapped Python functions.
5. Host appends tool outputs as `role="tool"` messages.
6. Host performs second model call.
7. Model returns final student-friendly explanation grounded in tool results.

### Q6) What errors can still happen even with function calling?

- Malformed user math syntax.
- Invalid or incomplete tool arguments.
- Numerical domain/runtime issues during plotting/evaluation.
- API/network failures.
- Model selecting an inappropriate tool in borderline prompts.

Function calling reduces error classes but does not eliminate them.

### Q7) When should the model answer directly vs. call a tool?

Direct answer is appropriate for conceptual or explanatory questions not requiring computation.  
Tool call is appropriate when correctness depends on calculation, symbolic solving, or plotting outputs.

---

## 6) Reflection: Model as Orchestrator vs Calculator

The strongest performance came from treating the model as an **orchestrator** rather than a raw calculator.

- As orchestrator, the model maps intent to tool usage decisions.
- As calculator, it would rely on probabilistic text generation for arithmetic/symbolic precision.
- The hybrid design separates concerns: language reasoning in the model, mathematical correctness in specialized libraries.

Educationally, this architecture is valuable because students can inspect each stage (request, tool call, tool output, final response), making agent behavior transparent rather than opaque.
