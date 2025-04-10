from pathlib import Path
from google.adk.tools.tool_context import ToolContext


def list_dir(path: str, tool_context: ToolContext) -> dict:
  root = tool_context.state["ox_ctx"]
  print(f"root={root}")
  print(f"list_dir(path={path})")

