# code-agent/tools.py

import pathlib
import logging
from typing import List, Dict, Any, Union

# Configure logging for tool operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileSystemTools:
    """Encapsulates file system operations restricted to a root directory."""

    def __init__(self, root_dir: pathlib.Path):
        """
        Initializes the tools with a specific root directory.

        Args:
            root_dir: The absolute path to the root directory for operations.
        """
        self.root_dir = root_dir.resolve() # Ensure absolute path
        logger.info(f"FileSystemTools initialized with root: {self.root_dir}")

    def _resolve_and_validate_path(self, target_path_str: str) -> Union[pathlib.Path, str]:
        """
        Resolves the target path relative to the root directory and validates it.

        Args:
            target_path_str: The relative path string provided by the user or model.

        Returns:
            A resolved pathlib.Path object if valid and within the root directory,
            otherwise an error message string.
        """
        try:
            # Prevent path traversal attacks by resolving carefully
            target_path = self.root_dir.joinpath(target_path_str).resolve()

            # Security Check: Ensure the resolved path is still within the root_dir
            if self.root_dir not in target_path.parents and target_path != self.root_dir:
                logger.warning(f"Attempted access outside root: {target_path_str} (resolved to {target_path})")
                return f"Error: Access denied. Path '{target_path_str}' is outside the allowed directory."

            return target_path
        except Exception as e:
            logger.error(f"Error resolving path '{target_path_str}': {e}", exc_info=True)
            return f"Error: Could not resolve path '{target_path_str}': {e}"

    def ls(self, path: str = ".") -> Union[str, List[str]]:
        """
        Lists the contents of a directory within the root directory.

        Args:
            path: The relative path to the directory to list (default is root).

        Returns:
            A list of filenames and directory names or an error string.
        """
        logger.info(f"Executing ls tool for path: '{path}'")
        resolved_path = self._resolve_and_validate_path(path)
        if isinstance(resolved_path, str): # Error occurred during validation
            return resolved_path

        if not resolved_path.exists():
            logger.warning(f"ls: Path does not exist: {resolved_path}")
            return f"Error: Path '{path}' does not exist."
        if not resolved_path.is_dir():
            logger.warning(f"ls: Path is not a directory: {resolved_path}")
            return f"Error: Path '{path}' is not a directory."

        try:
            contents = [item.name + ('/' if item.is_dir() else '') for item in resolved_path.iterdir()]
            logger.info(f"ls successful for '{path}'. Found {len(contents)} items.")
            return contents if contents else ["Directory is empty."]
        except PermissionError:
            logger.error(f"ls: Permission denied for {resolved_path}")
            return f"Error: Permission denied to list directory '{path}'."
        except Exception as e:
            logger.error(f"ls: Unexpected error for '{path}': {e}", exc_info=True)
            return f"Error: Could not list directory '{path}': {e}"

    def cat(self, file_path: str) -> str:
        """
        Reads the content of a file within the root directory.

        Args:
            file_path: The relative path to the file to read.

        Returns:
            The content of the file as a string or an error string.
        """
        logger.info(f"Executing cat tool for file: '{file_path}'")
        if not file_path:
             return "Error: No file path specified for cat command."

        resolved_path = self._resolve_and_validate_path(file_path)
        if isinstance(resolved_path, str): # Error occurred during validation
            return resolved_path

        if not resolved_path.exists():
            logger.warning(f"cat: File does not exist: {resolved_path}")
            return f"Error: File '{file_path}' does not exist."
        if not resolved_path.is_file():
            logger.warning(f"cat: Path is not a file: {resolved_path}")
            return f"Error: Path '{file_path}' is not a file."

        try:
            # Read file content, attempting different encodings if UTF-8 fails
            try:
                content = resolved_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                logger.warning(f"cat: UTF-8 decoding failed for {resolved_path}. Trying latin-1.")
                try:
                    content = resolved_path.read_text(encoding='latin-1')
                except Exception as inner_e:
                    logger.error(f"cat: Also failed decoding {resolved_path} with latin-1: {inner_e}")
                    return f"Error: Could not decode file '{file_path}' with UTF-8 or latin-1."
            # Optional: Add limit to content size if needed
            # MAX_CONTENT_SIZE = 10000
            # if len(content) > MAX_CONTENT_SIZE:
            #     content = content[:MAX_CONTENT_SIZE] + "\n... [content truncated]"
            logger.info(f"cat successful for '{file_path}'. Read {len(content)} characters.")
            return content
        except PermissionError:
            logger.error(f"cat: Permission denied for {resolved_path}")
            return f"Error: Permission denied to read file '{file_path}'."
        except Exception as e:
            logger.error(f"cat: Unexpected error for '{file_path}': {e}", exc_info=True)
            return f"Error: Could not read file '{file_path}': {e}"

# --- Gemini Tool Declarations ---
# These describe the tools to the Gemini model

LS_TOOL_DECLARATION = {
    "name": "ls",
    "description": "Lists the contents (files and directories) of a specified directory relative to the project root. Use '.' for the root directory itself.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The relative path to the directory to list. Defaults to '.' (the project root directory)."
            }
        },
         "required": [] # path is optional, defaults to '.'
    }
}

CAT_TOOL_DECLARATION = {
    "name": "cat",
    "description": "Reads and returns the content of a specified file relative to the project root.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The relative path to the file to read."
            }
        },
        "required": ["file_path"]
    }
}

# Combine tool declarations and implementations
AVAILABLE_TOOLS = {
    "ls": LS_TOOL_DECLARATION,
    "cat": CAT_TOOL_DECLARATION,
}

def get_tool_declarations() -> List[Dict[str, Any]]:
    """Returns the list of tool declarations for the Gemini model."""
    return list(AVAILABLE_TOOLS.values())

def get_tool_function(tool_name: str, fs_tools: FileSystemTools) -> callable:
    """Gets the callable function for a given tool name."""
    if tool_name == "ls":
        return fs_tools.ls
    elif tool_name == "cat":
        return fs_tools.cat
    else:
        raise ValueError(f"Unknown tool: {tool_name}")
