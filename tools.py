import shlex
import subprocess
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
  Reads the content of each file specified in `paths`.
  """
  pieces = [f"{p}\n\n{_read_file(p, tool_context)}" for p in paths]
  return "\n".join(pieces)


def find(path: str, pattern: str, tool_context: ToolContext) -> str:
  """
  Uses the command-line 'find' utility to recursively search for files and directories
  within the given directory whose names contain the specified substring (case-insensitive).

  The command is executed in the target directory so that the output paths are relative.

  Returns a newline separated list of matching paths or an informative message if no match is found.
  """
  try:
    p = _resolve_path(path, tool_context)
  except ValueError as e:
    return str(e)

  if not p.is_dir():
    return f"{path} is not a directory"

  # Build the find command.
  # Changing directory to `p` ensures output paths are relative.
  cmd_pattern = shlex.quote("*" + pattern + "*")
  cmd = f"find . -iname {cmd_pattern}"
  output = _run_cmd(cmd, cwd=str(p))

  if output:
    return output
  else:
    return f"No entries matching pattern '{pattern}' found in {path}"


def grep(pattern: str, paths: list[str], tool_context: ToolContext) -> str:
  """
  Uses the command-line 'grep' utility to search for lines matching a regular expression pattern
  within each of the specified files. The output is formatted as:
    <file>:<line_number>:<line_content>
  and includes 2 lines of context before and after each match.

  This updated version supports wildcard patterns in file paths (e.g. '*.txt') and limits
  the overall output size to avoid overflowing the LLM context.

  Returns a newline separated list of all matching lines or an informative message if no match is found.
  """
  MAX_GREP_OUTPUT = 4096  # maximum number of characters in the final grep output
  result_lines = []
  quoted_pattern = shlex.quote(pattern)

  for path_str in paths:
    # Check if the provided path string contains a wildcard character.
    if any(wc in path_str for wc in ['*', '?', '[']):
      # Expand the glob pattern.
      matched_files = _expand_glob_path(path_str, tool_context)
      if not matched_files:
        result_lines.append(f"{path_str} did not match any files")
        continue

      for file_path in matched_files:
        if not file_path.is_file():
          result_lines.append(f"{file_path} is not a file")
          continue
        cmd = f"grep -H -n -B2 -A2 {quoted_pattern} {shlex.quote(str(file_path))}"
        output = _run_cmd(cmd)
        if output:
          result_lines.append(output)
    else:
      # Process normally if no wildcard characters are present.
      try:
        file_path = _resolve_path(path_str, tool_context)
      except ValueError as e:
        result_lines.append(f"{path_str}: {e}")
        continue

      if not file_path.is_file():
        result_lines.append(f"{path_str} is not a file")
        continue

      cmd = f"grep -H -n -B2 -A2 {quoted_pattern} {shlex.quote(str(file_path))}"
      output = _run_cmd(cmd)
      if output:
        result_lines.append(output)

  combined_output = "\n".join(result_lines)
  if len(combined_output) > MAX_GREP_OUTPUT:
    combined_output = combined_output[:MAX_GREP_OUTPUT] + "\n[Output truncated]"
  if combined_output:
    return combined_output
  else:
    return f"No matches found for pattern '{pattern}'"


def _expand_glob_path(pattern: str, tool_context: ToolContext) -> list[Path]:
  """
  Expands a wildcard pattern into a list of matching Path objects.
  The pattern is resolved relative to the tool's root.
  """
  ctx: OxContext = tool_context.state[CTX_KEY]
  root = ctx.root

  # Create a Path from the pattern. If it's not absolute, treat it as relative to root.
  p = Path(pattern)
  if not p.is_absolute():
    p = (root / p)

  # Use glob to expand the wildcard pattern.
  # p.parent is the directory part and p.name is the glob pattern.
  try:
    matches = list(p.parent.glob(p.name))
  except Exception as e:
    raise ValueError(f"Invalid glob pattern: {pattern} ({e})")

  # Filter matches to include only those within the root.
  valid_matches = []
  for match in matches:
    try:
      match.relative_to(root)
      valid_matches.append(match.resolve())
    except ValueError:
      # The match is not within the allowed root.
      continue
  return valid_matches


class ToolBox:
  # Register all tools including list_dir, read_files, find, and grep.
  tools = [list_dir, read_files, find, grep]

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


def _run_cmd(cmd: str, cwd: str = None) -> str:
  try:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if result.stdout.strip():
      return result.stdout.strip()
    elif result.stderr.strip():
      return result.stderr.strip()
    else:
      return ""
  except Exception as e:
    return f"Error running command: {str(e)}"
