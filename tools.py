from pathlib import Path
from google.adk.tools.tool_context import ToolContext


def list_dir(path: str, tool_context: ToolContext) -> str:
    try:
        p = _resolve_path(path, tool_context)
    except ValueError as e:
        return str(e)

    if not p.is_dir():
        return f"{path} is not a directory"

    try:
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        if not entries:
            return f"{path} is an empty directory"

        return "\n".join(entry.name for entry in entries)
    except Exception as e:
        return f"Error reading directory: {str(e)}"


def read_file(path: str, tool_context: ToolContext) -> str:
    try:
        p = _resolve_path(path, tool_context)
    except ValueError as e:
        return str(e)

    if not p.is_file():
        return f"{path} is not a file"

    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"Failed to read file: {str(e)}"


def _resolve_path(path_str: str, tool_context: ToolContext) -> Path:
  root = Path(tool_context.state["ox_ctx"]).resolve()

  try:
    target = Path(path_str)
    if not target.is_absolute():
      target = (root / target).resolve()
    else:
      target = target.resolve()
  except Exception:
    raise ValueError(f"Invalid path: {path_str}")

  try:
    target.relative_to(root)
    return target
  except ValueError:
    raise ValueError(f"Path {path_str} is not relative to the root: {root}")
