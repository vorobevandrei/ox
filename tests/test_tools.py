import pytest
from pathlib import Path

from tools import find, grep, CTX_KEY


# Dummy classes to simulate the OxContext and ToolContext required by the commands.
class DummyOxContext:
  def __init__(self, root: Path):
    self.root = root


class DummyToolContext:
  def __init__(self, root: Path):
    # The state must include the CTX_KEY as expected by _resolve_path.
    self.state = {CTX_KEY: DummyOxContext(root)}


@pytest.fixture
def tool_context(tmp_path: Path) -> DummyToolContext:
  """
    Create a dummy tool context with the temporary directory as the root.
    """
  return DummyToolContext(tmp_path)


def test_find(tmp_path: Path, tool_context: DummyToolContext):
  """
    Test the 'find' command using a temporary directory structure.
    """
  # Create directory structure.
  dir1 = tmp_path / "dir1"
  dir1.mkdir()
  dir2 = tmp_path / "dir2"
  dir2.mkdir()

  # Create files with names to test matching on the substring "foo".
  file1 = dir1 / "foo.txt"
  file1.write_text("content1", encoding="utf-8")
  file2 = dir1 / "bar.txt"
  file2.write_text("content2", encoding="utf-8")
  file3 = dir2 / "foobar.log"
  file3.write_text("content3", encoding="utf-8")
  file4 = tmp_path / "other.txt"
  file4.write_text("content4", encoding="utf-8")

  # Run find with pattern "foo". The search should locate 'foo.txt' and 'foobar.log'.
  output = find(str(tmp_path), "foo", tool_context)
  lines = output.splitlines()

  # Verify that only files with "foo" in their name are present in the output.
  assert any("foo.txt" in line for line in lines), "Did not find 'foo.txt' in output."
  assert any("foobar.log" in line for line in lines), "Did not find 'foobar.log' in output."

  # Ensure that files without 'foo' (i.e. 'bar.txt' and 'other.txt') are not listed.
  assert all("bar.txt" not in line for line in lines), "'bar.txt' unexpectedly found in output."
  assert all("other.txt" not in line for line in lines), "'other.txt' unexpectedly found in output."


def test_grep_context(tmp_path: Path, tool_context: DummyToolContext):
  """
    Test the 'grep' command by creating a temporary file and checking
    that matching lines are output with the correct context (2 lines before and after).
    """
  # Create a file with content to test context output.
  test_file = tmp_path / "test.txt"
  content = "\n".join([
    "first line",  # context for match on line 2
    "second line with foo",  # match at line 2
    "third line",  # context after match at line 2
    "fourth line",  # context for match on line 5
    "fifth line with foo",  # match at line 5
    "sixth line",  # context after match at line 5
    "seventh line"  # further content
  ])
  test_file.write_text(content, encoding="utf-8")

  # Run grep for "foo" in the test file.
  output = grep("foo", [str(test_file)], tool_context)

  # The output should include the matched lines along with 2 context lines before and after each.
  assert "second line with foo" in output, "Expected match 'second line with foo' not found."
  assert "fifth line with foo" in output, "Expected match 'fifth line with foo' not found."
  # Check that some expected context lines appear.
  assert "first line" in output, "Expected context 'first line' not found."
  assert "third line" in output, "Expected context 'third line' not found."
  assert "fourth line" in output, "Expected context 'fourth line' not found."
  assert "sixth line" in output, "Expected context 'sixth line' not found."


def test_grep_wildcard(tmp_path: Path, tool_context: DummyToolContext):
  """
    Test the 'grep' command's wildcard support by creating multiple files and
    searching using a glob pattern.
    """
  # Create two files that match the glob pattern.
  file_a = tmp_path / "a.log"
  file_b = tmp_path / "b.log"
  file_a.write_text("alpha\nfoo\nomega", encoding="utf-8")
  file_b.write_text("beta\ngamma\ndelta", encoding="utf-8")

  # Use grep to search for the pattern "foo" with a wildcard pattern.
  output = grep("foo", [str(tmp_path / "*.log")], tool_context)

  # Expect to see file 'a.log' in the output since it contains the match.
  assert "a.log" in output, "Expected 'a.log' to be in the grep output."
  # Ensure that file 'b.log' does not appear because it does not include the match.
  assert "b.log" not in output, "File 'b.log' should not appear in grep output for pattern 'foo'."


def test_grep_truncation(tmp_path: Path, tool_context: DummyToolContext):
  """
    Test that the grep command output is truncated if it exceeds the maximum allowed length.
    """
  # Create a file with many matches to generate a long output.
  test_file = tmp_path / "big.txt"
  lines = []
  for i in range(1, 300):
    # Each block will generate multiple lines (match + context).
    lines.extend([
      f"Line {i} before",
      f"Line {i} with foo",
      f"Line {i} after",
      "separator"
    ])
  test_file.write_text("\n".join(lines), encoding="utf-8")

  output = grep("foo", [str(test_file)], tool_context)

  # Check that the output indicates that it has been truncated.
  assert "[Output truncated]" in output, "Expected '[Output truncated]' message not found."
  # Ensure that the output length does not exceed the maximum allowed (4096 characters plus truncation notice).
  max_length = 4096 + len("\n[Output truncated]")
  assert len(output) <= max_length, "Output exceeds the maximum allowed length."
