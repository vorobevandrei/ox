from dataclasses import dataclass
from pathlib import Path
from google.adk.tools.tool_context import ToolContext

from context import OxContext, CTX_KEY


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


def read_files(paths: list[str], tool_context: ToolContext) -> str:
  """
  Reads the content of each file specified in `paths`. Always prefer reading multiple files at once if possible.
  """
  pieces = [f"{p}\n\n{_read_file(p, tool_context)}" for p in paths]
  return "\n".join(pieces)


class ToolBox:
  tools = [list_dir, read_files]

  @classmethod
  def tools_names(cls) -> list[str]:
    return [tool.__name__ for tool in cls.tools]


def _read_file(path: str, tool_context: ToolContext) -> str:
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
  ctx: OxContext = tool_context.state[CTX_KEY]
  root = ctx.root

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
