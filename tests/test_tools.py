import pytest
from pathlib import Path

from tools import find, grep, tree, CTX_KEY, MAX_OUT_LENGTH


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
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    dir2 = tmp_path / "dir2"
    dir2.mkdir()

    file1 = dir1 / "foo.txt"
    file1.write_text("content1", encoding="utf-8")
    file2 = dir1 / "bar.txt"
    file2.write_text("content2", encoding="utf-8")
    file3 = dir2 / "foobar.log"
    file3.write_text("content3", encoding="utf-8")
    file4 = tmp_path / "other.txt"
    file4.write_text("content4", encoding="utf-8")

    output = find(str(tmp_path), "foo", tool_context)
    lines = output.splitlines()

    assert any("foo.txt" in line for line in lines), "Did not find 'foo.txt' in output."
    assert any("foobar.log" in line for line in lines), "Did not find 'foobar.log' in output."
    assert all("bar.txt" not in line for line in lines), "'bar.txt' unexpectedly found in output."
    assert all("other.txt" not in line for line in lines), "'other.txt' unexpectedly found in output."


def test_grep_context(tmp_path: Path, tool_context: DummyToolContext):
    """
    Test the 'grep' command by checking that matching lines are output with the expected context.
    """
    test_file = tmp_path / "test.txt"
    content = "\n".join([
        "first line",
        "second line with foo",
        "third line",
        "fourth line",
        "fifth line with foo",
        "sixth line",
        "seventh line"
    ])
    test_file.write_text(content, encoding="utf-8")

    output = grep("foo", [str(test_file)], tool_context)

    assert "second line with foo" in output, "Expected match 'second line with foo' not found."
    assert "fifth line with foo" in output, "Expected match 'fifth line with foo' not found."
    assert "first line" in output, "Expected context 'first line' not found."
    assert "third line" in output, "Expected context 'third line' not found."
    assert "fourth line" in output, "Expected context 'fourth line' not found."
    assert "sixth line" in output, "Expected context 'sixth line' not found."


def test_grep_wildcard(tmp_path: Path, tool_context: DummyToolContext):
    """
    Test the 'grep' command's wildcard support.
    """
    file_a = tmp_path / "a.log"
    file_b = tmp_path / "b.log"
    file_a.write_text("alpha\nfoo\nomega", encoding="utf-8")
    file_b.write_text("beta\ngamma\ndelta", encoding="utf-8")

    output = grep("foo", [str(tmp_path / "*.log")], tool_context)

    assert "a.log" in output, "Expected 'a.log' to be in the grep output."
    assert "b.log" not in output, "File 'b.log' should not appear in grep output for pattern 'foo'."


def test_grep_truncation(tmp_path: Path, tool_context: DummyToolContext):
    """
    Test that the grep command output is truncated if it exceeds the maximum allowed length.
    """
    test_file = tmp_path / "big.txt"
    lines = []
    for i in range(1, 300):
        lines.extend([
            f"Line {i} before",
            f"Line {i} with foo",
            f"Line {i} after",
            "separator"
        ])
    test_file.write_text("\n".join(lines), encoding="utf-8")

    output = grep("foo", [str(test_file)], tool_context)

    assert "[Output truncated]" in output, "Expected '[Output truncated]' message not found."
    assert len(output) <= MAX_OUT_LENGTH, "Output exceeds the maximum allowed length."


def test_tree_directory_only(tmp_path: Path, tool_context: DummyToolContext):
    """
    Test the 'tree' command when listing only directories.
    """
    testdir = tmp_path / "testdir"
    testdir.mkdir()
    subdir = testdir / "subdir"
    subdir.mkdir()
    file_in_testdir = testdir / "file.txt"
    file_in_testdir.write_text("dummy", encoding="utf-8")

    output = tree(str(testdir), tool_context, max_depth=2, include_files=False)
    if output.startswith("The 'tree' command is not available"):
        pytest.skip("tree command not available on this system")

    assert "subdir" in output, "Directory 'subdir' should appear in tree output."
    assert "file.txt" not in output, "File 'file.txt' should not appear when include_files is False."


def test_tree_include_files(tmp_path: Path, tool_context: DummyToolContext):
    """
    Test the 'tree' command when including files.
    """
    testdir = tmp_path / "testdir"
    testdir.mkdir()
    subdir = testdir / "subdir"
    subdir.mkdir()
    file_in_testdir = testdir / "file.txt"
    file_in_testdir.write_text("dummy", encoding="utf-8")
    file_in_subdir = subdir / "inner.txt"
    file_in_subdir.write_text("dummy", encoding="utf-8")

    output = tree(str(testdir), tool_context, max_depth=3, include_files=True)
    if output.startswith("The 'tree' command is not available"):
        pytest.skip("tree command not available on this system")

    assert "file.txt" in output, "File 'file.txt' should appear when include_files is True."
    assert "inner.txt" in output, "File 'inner.txt' should appear when include_files is True."


def test_tree_max_depth(tmp_path: Path, tool_context: DummyToolContext):
    """
    Test that the tree output respects the max_depth parameter.
    Instead of just checking for the absence of a specific name,
    we verify that no directory entry line has an indent level exceeding max_depth-2.
    """
    lvl1 = tmp_path / "level1"
    lvl1.mkdir()
    lvl2 = lvl1 / "level2"
    lvl2.mkdir()
    lvl3 = lvl2 / "level3"
    lvl3.mkdir()

    max_depth = 2
    output = tree(str(lvl1), tool_context, max_depth=max_depth, include_files=True)
    if output.startswith("The 'tree' command is not available"):
        pytest.skip("tree command not available on this system")

    # Check that at least level2 is present.
    assert "level2" in output, "Directory 'level2' should appear in tree output."

    # Verify no line corresponding to a directory entry has indent > (max_depth - 2).
    # For max_depth=2, allowed indent groups = 0.
    allowed_indent = max_depth - 2
    for line in output.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("└──") or stripped.startswith("├──"):
            indent_level = (len(line) - len(line.lstrip(" "))) // 4
            assert indent_level <= allowed_indent, f"Indent level {indent_level} exceeds allowed {allowed_indent}"


def test_tree_not_directory(tmp_path: Path, tool_context: DummyToolContext):
    """
    Test that tree returns an error message when the path is not a directory.
    """
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("dummy", encoding="utf-8")

    output = tree(str(file_path), tool_context)
    assert "is not a directory" in output, "Expected error message for non-directory input."


def test_tree_truncation(tmp_path: Path, tool_context: DummyToolContext):
    """
    Test that the tree command output is truncated if it exceeds the maximum allowed length.
    """
    big = tmp_path / "big"
    big.mkdir()
    for i in range(10):
        sub = big / f"subdir_{i}"
        sub.mkdir()
        for j in range(5):
            nested = sub / f"nested_{j}"
            nested.mkdir()

    max_output = 100
    output = tree(str(big), tool_context, max_depth=3, include_files=True, max_output=max_output)
    if output.startswith("The 'tree' command is not available"):
        pytest.skip("tree command not available on this system")

    assert "[Output truncated]" in output, "Expected output to be truncated when exceeding max_output limit."
    assert len(output) <= max_output, "Output length should not exceed the max_output limit."
