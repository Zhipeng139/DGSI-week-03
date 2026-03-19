import os
import json
import sqlite3
import subprocess
from dotenv import load_dotenv
from openai import OpenAI

# UI Imports
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich.align import Align
from rich.markdown import Markdown
from rich.rule import Rule
from rich import box

# Load credentials
load_dotenv()
console = Console()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'), 
    base_url=os.getenv('OPENAI_API_ENDPOINT')
)
MODEL = os.getenv('MODEL', 'gpt-4.5-preview')

# ============================================================================
# TOOLS
# ============================================================================

def execute_sql(query: str) -> str:
    """Executes a SQL query and returns results formatted for the LLM."""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute(query)
        
        if query.strip().upper().startswith('SELECT'):
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            conn.close()
            return json.dumps({"columns": columns, "data": rows})
        
        conn.commit()
        conn.close()
        return "Query executed successfully"
    except sqlite3.Error as e:
        return f"Database error: {e}"

def tool_wget(url: str, flags: str = "-q -O -") -> str:
    """Fetches a URL using wget with user confirmation."""
    cmd = f"wget {flags} {url}"
    
    console.print(Panel(
        Text(f"The LLM wants to run:\n{cmd}", style="bold yellow"),
        title="[bold red]Security Warning[/bold red]",
        subtitle="User confirmation required",
        border_style="yellow",
        expand=False
    ))
    
    answer = console.input("[bold cyan]Allow execution? (y/n): [/bold cyan]")
    if answer.lower() != "y":
        return "USER DENIED: command was not executed."
    
    try:
        result = subprocess.run(
            ["wget", "-q", "-O", "-", url],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Command failed: {e.stderr}"
    except Exception as e:
         return f"Error: {e}"

# ============================================================================
# SCHEMAS & DISPATCH
# ============================================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Run a SQL statement against the local SQLite database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The SQL statement to execute."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wget",
            "description": "Fetches a URL using the system wget command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch."}
                },
                "required": ["url"]
            }
        }
    }
]

dispatch = {"execute_sql": execute_sql, "wget": tool_wget}

# ============================================================================
# UI HELPERS
# ============================================================================

def print_sql_table(json_data: str):
    """Prints a beautiful table for SELECT results."""
    try:
        data = json.loads(json_data)
        if not isinstance(data, dict) or "columns" not in data:
            console.print(Panel(Text(json_data, style="italic"), border_style="blue"))
            return

        table = Table(box=box.MINIMAL_DOUBLE_HEAD, border_style="blue", show_header=True, header_style="bold cyan")
        for col in data["columns"]:
            table.add_column(col)
        
        for row in data["data"]:
            table.add_row(*[str(item) for item in row])
        
        console.print(table)
    except:
        console.print(Panel(Text(json_data, style="italic"), border_style="blue"))

# ============================================================================
# MAIN LOOP
# ============================================================================

def run_chat(prompt: str):
    messages = [{"role": "user", "content": prompt}]
    
    while True:
        with Live(Spinner("dots", text="[bold yellow]Agent is thinking...[/bold yellow]"), refresh_per_second=10, transient=True):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
            )
        
        message = response.choices[0].message
        
        # UI: Show assistant's text content if present (even with tool calls)
        if message.content:
            console.print(Panel(Markdown(message.content), title="[bold green]Assistant[/bold green]", border_style="green", padding=(1, 2)))
        
        if not message.tool_calls:
            break
            
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                } for tc in message.tool_calls
            ],
        })
        
        for tc in message.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            
            # UI: Cleaner Tool Call display
            console.print(Rule(f"[bold yellow]Tool Execution: {name}[/bold yellow]", style="yellow"))
            
            if "query" in args:
                console.print(Panel(Syntax(args["query"], "sql", theme="monokai", background_color="default"), title="SQL Query", border_style="cyan"))
            elif "url" in args:
                console.print(Panel(Text(args['url'], style="link"), title="Target URL", border_style="cyan"))
            
            result = dispatch[name](**args)
            
            # UI: Show result preview or table
            if name == "execute_sql" and "SELECT" in args.get("query", "").upper():
                print_sql_table(result)
            else:
                preview = str(result)[:800] + "..." if len(str(result)) > 800 else str(result)
                console.print(Panel(preview, title="Output", border_style="dim", subtitle=f"{name} returned {len(str(result))} chars"))
            
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": str(result),
            })

if __name__ == "__main__":
    console.clear()
    console.print(Align.center(Panel(
        Text("DATABASE AGENT LOOP\n", style="bold white") + Text("SQL & Wget Smart Interface", style="italic cyan"),
        border_style="bright_blue",
        padding=(1, 10),
        expand=False
    )))
    console.print(Align.center("[dim]Type 'exit' or 'quit' to end the session[/dim]\n"))
    
    while True:
         try:
             prompt = console.input("[bold blue]User ❯ [/bold blue]").strip()
             
             if not prompt:
                 continue
                 
             if prompt.lower() in ['exit', 'quit']:
                 console.print("\n[bold magenta]Goodbye! 👋[/bold magenta]")
                 break
             
             console.print(Rule(style="dim"))
             run_chat(prompt)
             console.print(Rule(style="dim"))
             
         except KeyboardInterrupt:
             console.print("\n[bold red]Session interrupted.[/bold red]")
             break
         except Exception as e:
             console.print(f"\n[bold red]CRITICAL ERROR:[/bold red] {e}")
