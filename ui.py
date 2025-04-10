# code-agent/ui.py

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.syntax import Syntax
import json

# Global console object
console = Console()

def display_message(role: str, content: str, is_markdown: bool = False) -> None:
    """Displays a message in the console with appropriate styling."""
    color = "blue"
    prefix = "[bold blue]Agent[/]"
    if role == "user":
        color = "green"
        prefix = "[bold green]You[/]"
    elif role == "tool_call":
        color = "yellow"
        prefix = "[bold yellow]Tool Call[/]"
        # Pretty print args if possible
        try:
            args = json.loads(content) # Assuming content is JSON string of args
            content_display = json.dumps(args, indent=2)
            content_display = Syntax(content_display, "json", theme="default", line_numbers=False)
        except json.JSONDecodeError:
             content_display = content # Fallback to raw string
    elif role == "tool_result":
        color = "magenta"
        prefix = "[bold magenta]Tool Result[/]"
        # Try basic formatting for lists
        if isinstance(content, list):
             content_display = "\n".join(f"- {item}" for item in content)
        else:
            content_display = str(content) # Ensure it's a string
    else: # Agent's final text response
        content_display = Markdown(content) if is_markdown else content

    console.print(Panel(content_display, title=prefix, border_style=color, expand=False))

def get_user_input() -> str:
    """Prompts the user for input."""
    return Prompt.ask("[bold green]You[/]> ")

def display_welcome(root_dir: str) -> None:
    """Displays a welcome message."""
    console.print(Panel(f"Welcome to the Code Agent!\nI can answer questions about the code in: [cyan]{root_dir}[/cyan]\nType 'exit' or 'quit' to end the session.",
                        title="Code Agent", border_style="bold white", expand=False))

def display_exit() -> None:
    """Displays an exit message."""
    console.print("[bold red]Exiting Code Agent. Goodbye![/]")

def display_error(message: str) -> None:
    """Displays an error message."""
    console.print(Panel(f"[bold red]Error:[/bold red] {message}", title="Error", border_style="red", expand=False))

def display_status(message: str):
    """Displays a temporary status message (e.g., 'Thinking...')"""
    return console.status(f"[bold yellow]{message}[/]")
