# code-agent/agent.py

from google import genai
import pathlib
import logging
from typing import Dict, Any, Generator, Tuple, Union, List
from google.genai.types import Content, Part, FunctionCall, FunctionResponse
import json
from tools import FileSystemTools, get_tool_declarations, get_tool_function

logger = logging.getLogger(__name__)

# Define message types for the generator
MessageType = Tuple[str, Union[str, List[str], Dict[str, Any]]] # (role, content)

class Agent:
    """Handles interaction with the Gemini model and tools."""

    def __init__(self, api_key: str, root_dir: pathlib.Path, model_name: str = "gemini-1.5-pro-latest"):
        """
        Initializes the agent.

        Args:
            api_key: The Google AI Studio API key.
            root_dir: The root directory for file system operations.
            model_name: The Gemini model to use (needs function calling).
        """
        logger.info(f"Initializing Agent with model: {model_name}")
        self.root_dir = root_dir
        self.fs_tools = FileSystemTools(root_dir)

        # Configure the SDK
        genai.configure(api_key=api_key)

        # Define tools for the model
        self.tool_declarations = get_tool_declarations()
        self.tool_config = {"function_calling_config": {"mode": "auto"}} # Let the model decide

        # System instruction to guide the agent
        system_instruction = f"""You are an AI assistant specialized in analyzing and answering questions about code and files within the directory '{root_dir.name}'.
You have access to two tools:
1. `ls(path)`: Lists files and directories at the given relative path. Use '.' for the root.
2. `cat(file_path)`: Reads the content of the specified relative file path.

Use these tools whenever necessary to examine the directory structure or file contents before answering the user's query.
Always provide paths relative to the root directory '{root_dir.name}'.
Be concise and helpful in your answers.
If a tool call results in an error (e.g., file not found, permission denied), inform the user about the error.
Do not attempt to access paths outside the root directory.
"""

        # Initialize the Gemini model
        self.model = genai.GenerativeModel(
            model_name=model_name,
            tools=self.tool_declarations,
            tool_config=self.tool_config,
            system_instruction=system_instruction
        )

        # Start a chat session
        self.chat = self.model.start_chat(enable_automatic_function_calling=False) # Manual control needed for UI feedback
        logger.info("Agent initialized and chat session started.")

    def _execute_tool_call(self, function_call: FunctionCall) -> FunctionResponse:
        """Executes a function call requested by the model."""
        tool_name = function_call.name
        args = dict(function_call.args) # Convert Struct to dict
        logger.info(f"Attempting to execute tool: {tool_name} with args: {args}")

        try:
            tool_function = get_tool_function(tool_name, self.fs_tools)
            # Ensure required args are present (handle Gemini sometimes omitting optional args)
            if tool_name == 'ls' and 'path' not in args:
                args['path'] = '.' # Default for ls

            result = tool_function(**args)
            logger.info(f"Tool '{tool_name}' executed successfully.")
            # Ensure result is serializable (Gemini expects JSON-like structure)
            if isinstance(result, list):
                serializable_result = {"result": result}
            elif isinstance(result, str):
                 serializable_result = {"result": result} # Wrap strings too
            else:
                 serializable_result = {"result": str(result)} # Fallback

            return Part.from_function_response(
                name=tool_name,
                response=serializable_result
            )
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
            return Part.from_function_response(
                name=tool_name,
                response={
                    "error": f"Failed to execute tool '{tool_name}' with args {args}. Error: {e}"
                }
            )

    def send_message(self, user_input: str) -> Generator[MessageType, None, None]:
        """
        Sends user input to the model and handles the conversation flow,
        including tool calls and responses, yielding messages for the UI.

        Args:
            user_input: The text input from the user.

        Yields:
            Tuples of (role, content) for displaying in the UI.
            Roles can be 'user', 'tool_call', 'tool_result', 'agent'.
        """
        if not user_input:
            return

        yield "user", user_input
        current_input = user_input

        try:
            while True:
                # Send message or tool response to the model
                response = self.chat.send_message(current_input)
                candidate = response.candidates[0]

                # Check for function calls
                if candidate.content.parts and isinstance(candidate.content.parts[0], FunctionCall):
                    function_call = candidate.content.parts[0].function_call
                    tool_name = function_call.name
                    tool_args = dict(function_call.args)

                    logger.info(f"Model requested tool call: {tool_name}({tool_args})")
                    yield "tool_call", f"{tool_name}({json.dumps(tool_args)})" # Yield call info

                    # Execute the tool call
                    api_tool_response = self._execute_tool_call(function_call)
                    tool_result_content = api_tool_response.function_response.response # Extract result data
                    logger.info(f"Tool '{tool_name}' response: {tool_result_content}")
                    yield "tool_result", tool_result_content # Yield result

                    # Send the tool response back to the model
                    current_input = api_tool_response # Next input is the tool result Part

                # Check for final text response
                elif candidate.content.parts and candidate.content.parts[0].text:
                    final_response = candidate.text
                    logger.info(f"Model provided final text response.")
                    yield "agent", final_response
                    break # Exit loop once we have the final answer

                # Handle potential unexpected states (e.g., empty response)
                elif candidate.finish_reason != "STOP":
                     logger.warning(f"Model stopped unexpectedly. Finish Reason: {candidate.finish_reason}, Safety Ratings: {candidate.safety_ratings}")
                     yield "agent", f"[Agent Note: Model stopped unexpectedly. Reason: {candidate.finish_reason}]"
                     break
                else:
                     logger.warning("Received an empty or unexpected response part from the model.")
                     yield "agent", "[Agent Note: Received an empty response from the model.]"
                     break # Avoid infinite loop

        except Exception as e:
            logger.error(f"Error during chat interaction: {e}", exc_info=True)
            yield "agent", f"[Agent Error: An unexpected error occurred during processing: {e}]"
