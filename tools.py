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
  the overall output size to avoid overflowing.

  Returns a newline separated list of all matching lines or an informative message if no match is found.
  """
  MAX_GREP_OUTPUT = 4096  # maximum number of characters in the final grep output
  result_lines = []
  quoted_pattern = shlex.quote(pattern)

  for path_str in paths:
    if any(wc in path_str for wc in ['*', '?', '[']):
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
    suffix = "\n[Output truncated]"
    combined_output = combined_output[:MAX_GREP_OUTPUT - len(suffix)] + suffix
  if combined_output:
    return combined_output
  else:
    return f"No matches found for pattern '{pattern}'"


def tree(path: str, tool_context: ToolContext, max_depth: int = 1, include_files: bool = False,
         max_output: int = 4096) -> str:
  """
  Uses the system's 'tree' command to print a tree-like directory structure starting at the given path.

  Args:
      path: The directory path from which to start listing.
      tool_context: The ToolContext used for resolving paths.
      max_depth: Maximum depth (counting the starting directory as level 1) to display.
                 For example, if max_depth is 2, only the starting directory and its immediate children will be shown.
      include_files: If False, only directories are shown; if True, both files and directories are listed.
      max_output: Maximum number of characters allowed in the final output.

  Returns:
      A string representing the tree structure or an error message.
  """
  try:
    p = _resolve_path(path, tool_context)
  except ValueError as e:
    return str(e)

  if not p.is_dir():
    return f"{path} is not a directory"

  if max_depth < 1:
    return "max_depth must be at least 1"

  # Build the tree command.
  cmd = ["tree", "-L", str(max_depth)]
  if not include_files:
    cmd.append("-d")
  cmd.append(str(p))

  try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    raw_output = result.stdout if result.stdout else result.stderr
  except FileNotFoundError:
    return "The 'tree' command is not available on this system."
  except Exception as e:
    return f"Error running tree command: {str(e)}"

  # Post-process the output to enforce the intended max_depth.
  # In many tree command implementations the starting directory is not counted in -L.
  # To achieve the effect that max_depth==2 means "show only the starting directory and its immediate children",
  # we remove any lines (entries) whose indent level exceeds (max_depth - 2).
  lines = raw_output.splitlines()
  # If the first line equals the starting directory path, keep it as header.
  if lines and lines[0].strip() == str(p):
    header = lines[0]
    remaining_lines = lines[1:]
  else:
    header = ""
    remaining_lines = lines

  allowed_indent = max_depth - 2  # For max_depth==2, allowed_indent==0 (i.e. no grandchildren).
  filtered_lines = []
  for line in remaining_lines:
    stripped = line.lstrip()
    if stripped.startswith("└──") or stripped.startswith("├──"):
      indent_level = (len(line) - len(line.lstrip(" "))) // 4
      if indent_level <= allowed_indent:
        filtered_lines.append(line)
    else:
      # Keep summary lines and any non-connector lines.
      filtered_lines.append(line)

  output = (header + "\n" + "\n".join(filtered_lines)) if header else "\n".join(filtered_lines)

  # Enforce maximum output size.
  if len(output) > max_output:
    notice = "\n[Output truncated]"
    allowed = max_output - len(notice)
    output = output[:allowed] + notice
  return output


def _expand_glob_path(pattern: str, tool_context: ToolContext) -> list[Path]:
  """
  Expands a wildcard pattern into a list of matching Path objects.
  The pattern is resolved relative to the tool's root.
  """
  ctx: OxContext = tool_context.state[CTX_KEY]
  root = ctx.root

  p = Path(pattern)
  if not p.is_absolute():
    p = root / p

  try:
    matches = list(p.parent.glob(p.name))
  except Exception as e:
    raise ValueError(f"Invalid glob pattern: {pattern} ({e})")

  valid_matches = []
  for match in matches:
    try:
      match.relative_to(root)
      valid_matches.append(match.resolve())
    except ValueError:
      continue
  return valid_matches


class ToolBox:
  # Register all tools including list_dir, read_files, find, grep, and tree.
  tools = [list_dir, read_files, find, grep, tree]

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
