import os
import json
import pathlib
import tempfile
import shutil
from typing import List, Dict, Any, Generator, Optional

from google import genai
from google.genai import types
from google.protobuf import json_format # For converting proto args to dict

# --- Tool Interfaces and Implementations ---

class Tool:
    """Base class for tools"""
    name: str
    description: str
    schema: types.Schema

    def get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=self.schema,
        )

    def execute(self, **kwargs) -> str:
        """Executes the tool with given arguments."""
        raise NotImplementedError


class LsTool(Tool):
    """Tool to list directory contents."""
    name = "ls"
    description = "Lists the contents (files and directories) of a specified directory relative to the agent's current working directory. If no path is provided, lists the contents of the current working directory."
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "path": types.Schema(
                type=types.Type.STRING,
                description="Optional path relative to the current directory. Defaults to '.' if omitted.",
                nullable=True # Indicate path is optional
            ),
        },
        # No 'required' list means 'path' is optional by default
    )

    def __init__(self, working_directory: pathlib.Path):
        self._working_directory = working_directory.resolve() # Ensure absolute path

    def execute(self, path: Optional[str] = None) -> str:
        """Lists files and directories in the specified path relative to the working directory."""
        try:
            if path:
                # Resolve relative to working directory
                target_path = (self._working_directory / path).resolve()
                # Security check: Ensure the target path is still within the original working directory
                if self._working_directory not in target_path.parents and target_path != self._working_directory:
                     return f"Error: Access denied. Path '{path}' is outside the allowed directory."
            else:
                target_path = self._working_directory

            if not target_path.exists():
                return f"Error: Path does not exist: {path or '.'}"
            if not target_path.is_dir():
                return f"Error: Path is not a directory: {path or '.'}"

            contents = []
            for item in sorted(target_path.iterdir()):
                contents.append(f"{item.name}{'/' if item.is_dir() else ''}")

            if not contents:
                return f"Directory '{path or '.'}' is empty."
            else:
                # Add path context to the output
                relative_display_path = path if path else '.'
                return f"Contents of '{relative_display_path}':\n" + "\n".join(contents)

        except Exception as e:
            return f"Error executing ls on path '{path or '.'}': {e}"


class CatTool(Tool):
    """Tool to read file contents."""
    name = "cat"
    description = "Reads and returns the content of a specified file relative to the agent's current working directory."
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "filename": types.Schema(
                type=types.Type.STRING,
                description="The path to the file relative to the current directory.",
            ),
        },
        required=["filename"]
    )

    def __init__(self, working_directory: pathlib.Path):
        self._working_directory = working_directory.resolve() # Ensure absolute path

    def execute(self, filename: str) -> str:
        """Reads the content of the specified file relative to the working directory."""
        if not filename:
            return "Error: No filename provided for cat command."
        try:
            target_path = (self._working_directory / filename).resolve()

            # Security check: Ensure the target path is still within the original working directory
            if self._working_directory not in target_path.parents and target_path != self._working_directory:
                 return f"Error: Access denied. File '{filename}' is outside the allowed directory."

            if not target_path.exists():
                return f"Error: File not found: {filename}"
            if not target_path.is_file():
                return f"Error: Path is not a file: {filename}"

            # Read file content, handle potential encoding issues
            try:
                content = target_path.read_text(encoding='utf-8')
                # Optional: Add truncation for very large files
                max_len = 2000
                if len(content) > max_len:
                    content = content[:max_len] + f"\n... [truncated at {max_len} chars]"
                return f"Content of '{filename}':\n---\n{content}\n---"
            except UnicodeDecodeError:
                return f"Error: Cannot decode file '{filename}' as UTF-8 text. It might be a binary file."
            except Exception as e:
                 return f"Error reading file '{filename}': {e}"

        except Exception as e:
            return f"Error executing cat on file '{filename}': {e}"

# --- Agent Implementation ---

class FileSystemAgent:
    """An agent that interacts with the Gemini API and can use filesystem tools (ls, cat)."""

    def __init__(
        self,
        api_key: str,
        working_directory: pathlib.Path,
        model_name: str = "models/gemini-1.5-flash", # Use a model supporting function calling well
        temperature: float = 0.3,
    ):
        """
        Initializes the FileSystemAgent.

        Args:
            api_key: The Gemini API key.
            working_directory: The root directory the agent can operate within.
            model_name: The name of the Gemini model to use.
            temperature: The sampling temperature for the LLM.
        """
        if not working_directory.is_dir():
            raise ValueError(f"Working directory '{working_directory}' does not exist or is not a directory.")

        self.working_directory = working_directory.resolve()
        self.client = genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
             # System instructions can guide the model on how to use tools and its persona
            system_instruction=f"You are a helpful assistant that can interact with a file system within a specific directory: '{self.working_directory.name}'. You have access to 'ls' to list directory contents and 'cat' to read files. All paths provided to tools must be relative to this starting directory. Always confirm file/directory existence with 'ls' before trying to 'cat' a file if unsure.",
        )
        self.temperature = temperature
        self.history: List[types.Content] = []

        # Initialize tools
        self.ls_tool = LsTool(self.working_directory)
        self.cat_tool = CatTool(self.working_directory)
        self.tools_map: Dict[str, Tool] = {
            self.ls_tool.name: self.ls_tool,
            self.cat_tool.name: self.cat_tool,
        }

        # Prepare tool configuration for the API
        self.api_tools = types.Tool(
            function_declarations=[
                self.ls_tool.get_declaration(),
                self.cat_tool.get_declaration(),
            ]
        )
        self.generation_config = types.GenerationConfig(
            temperature=self.temperature,
            # response_mime_type="text/plain" # Not needed for chat usually
        )
        self.safety_settings = { # Adjust safety settings if needed
            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            types.HarmCategory.HARM_CATEGORY_HARASSMENT: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            # Add others if required
        }

        # Start a chat session (manages history internally for convenience)
        self.chat = self.model.start_chat(history=self.history, enable_automatic_function_calling=False) # Manual control


    def _execute_function_call(self, function_call: types.FunctionCall) -> types.Content:
        """Executes a function call and returns the result as API Content."""
        tool_name = function_call.name
        tool_args_dict = json_format.MessageToDict(function_call.args) # Convert proto Struct to Python dict

        if tool_name in self.tools_map:
            tool_instance = self.tools_map[tool_name]
            try:
                # Execute the tool's method
                result_str = tool_instance.execute(**tool_args_dict)
            except Exception as e:
                # Catch potential errors during execution (e.g., unexpected args)
                result_str = f"Error executing tool {tool_name}: {e}"
        else:
            result_str = f"Error: Unknown tool '{tool_name}' requested."

        # Format the result for the API
        return types.Content(
            role="function", # Role should be 'function' for a response
            parts=[
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result_str} # API expects a dict response
                )
            ]
        )

    def step(self, user_input: str) -> Generator[Dict[str, Any], None, None]:
        """
        Sends user input to the LLM, handles function calls, and yields results.

        Args:
            user_input: The text input from the user.

        Yields:
            Dictionaries representing streamed messages:
            - {"type": "text", "content": "..."} for text chunks.
            - {"type": "function_call", "name": "...", "args": {...}} for function calls initiated by the LLM.
            - {"type": "function_result", "name": "...", "result": "..."} for results from executed functions.
             - {"type": "error", "message": "..."} for agent-level errors.
        """
        print(f"\n[Agent] Processing user input: {user_input}")
        # Add user message explicitly to history (though start_chat also does this)
        # self.history.append(types.Content(role="user", parts=[types.Part.from_text(user_input)]))

        try:
            # Send message using the chat session
            response_stream = self.chat.send_message(
                user_input,
                stream=True,
                generation_config=self.generation_config,
                tools=[self.api_tools], # Pass tools on each turn for context
                safety_settings=self.safety_settings,
            )

            full_llm_text_response = "" # Accumulate text response

            for chunk in response_stream:
                # Check for function calls *first*
                if chunk.function_calls:
                    fc = chunk.function_calls[0] # Assuming one call per chunk for simplicity
                    args_dict = json_format.MessageToDict(fc.args)
                    print(f"[Agent] LLM requests function call: {fc.name}({args_dict})")
                    yield {"type": "function_call", "name": fc.name, "args": args_dict}

                    # Execute the function
                    function_response_content = self._execute_function_call(fc)
                    tool_result_str = function_response_content.parts[0].function_response.response['result'] # Extract result string
                    print(f"[Agent] Function result: {tool_result_str[:100]}...") # Log truncated result
                    yield {"type": "function_result", "name": fc.name, "result": tool_result_str }

                    # Send the function response back to the model *in a separate call within the loop*
                    # The chat session handles history, so just send the function response content
                    response_stream_after_func = self.chat.send_message(
                        function_response_content, # Send the result back
                        stream=True,
                        generation_config=self.generation_config,
                        # No need to send tools again when just providing func result
                        safety_settings=self.safety_settings,
                    )

                    # Process the *second* stream (response *after* function execution)
                    for final_chunk in response_stream_after_func:
                         if final_chunk.text:
                             print(f"[Agent] LLM text chunk (after func): {final_chunk.text}")
                             yield {"type": "text", "content": final_chunk.text}
                             full_llm_text_response += final_chunk.text
                         # Handle potential errors/safety issues in the second response if needed
                         if final_chunk.prompt_feedback.block_reason:
                             msg = f"Response blocked: {final_chunk.prompt_feedback.block_reason}"
                             print(f"[Agent] {msg}")
                             yield {"type": "error", "message": msg}
                             # Decide how to proceed, maybe break or yield error
                             break

                # If no function call, process text
                elif chunk.text:
                    print(f"[Agent] LLM text chunk: {chunk.text}")
                    yield {"type": "text", "content": chunk.text}
                    full_llm_text_response += chunk.text

                # Handle potential errors/safety issues in the first response
                if chunk.prompt_feedback.block_reason:
                     msg = f"Response blocked: {chunk.prompt_feedback.block_reason}"
                     print(f"[Agent] {msg}")
                     yield {"type": "error", "message": msg}
                     # Decide how to proceed
                     break


            # Update history manually *after* the turn if needed,
            # although chat session should handle most of it.
            # Let's ensure the final aggregated text response is added if it exists.
            # The ChatSession history already contains user, model (w/ func call), function response, model (final text)
            # print("\n[Agent] Final History:")
            # for content in self.chat.history:
            #     print(f"  Role: {content.role}")
            #     for part in content.parts:
            #          print(f"    Part: {part}")


        except Exception as e:
            error_message = f"An error occurred during agent step: {e}"
            print(f"[Agent] Error: {error_message}")
            yield {"type": "error", "message": error_message}
            # Optionally re-raise or handle differently
            # raise e


# --- Example Usage ---

if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        exit(1)

    # Create a temporary directory for the agent to work in
    temp_dir = None
    try:
        temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="agent_test_"))
        print(f"Created temporary working directory: {temp_dir}")

        # Create some dummy files and directories
        (temp_dir / "subdir").mkdir()
        (temp_dir / "hello.txt").write_text("This is the content of hello.txt.")
        (temp_dir / "subdir" / "data.json").write_text('{"key": "value", "number": 123}')
        (temp_dir / "another file.txt").write_text("File with spaces in its name.")
        # Create a binary file (e.g., small image - replace with actual data if needed)
        # Example: create small binary data
        binary_data = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) # PNG header start
        (temp_dir / "image.png").write_bytes(binary_data)

        print("\n--- Initializing Agent ---")
        agent = FileSystemAgent(api_key=api_key, working_directory=temp_dir)
        print("--- Agent Initialized ---")

        # --- Interaction Loop ---
        prompts = [
            "Hi there! What files are in the current directory?",
            "Can you show me the content of hello.txt?",
            "What about the file in the subdirectory?", # Requires ls subdir first potentially
            "Okay, show me the content of 'subdir/data.json'",
            "Try reading image.png",
            "What about 'another file.txt'?"
        ]

        for i, prompt in enumerate(prompts):
            print(f"\n--- Turn {i+1} ---")
            print(f"> User: {prompt}")
            print("< Agent:")

            output_buffer = ""
            try:
                for message in agent.step(prompt):
                    if message["type"] == "text":
                        print(message["content"], end="", flush=True)
                        output_buffer += message["content"]
                    elif message["type"] == "function_call":
                        print(f"\n   [Function Call: {message['name']}({message.get('args', {})})]")
                    elif message["type"] == "function_result":
                         # Optionally truncate long results for display
                         result_str = message.get('result', '')
                         display_result = result_str[:200] + ('...' if len(result_str) > 200 else '')
                         print(f"   [Function Result ({message['name']}): {display_result}]")
                    elif message["type"] == "error":
                        print(f"\n   [Error: {message['message']}]")
                        output_buffer += f"\n[Agent Error: {message['message']}]" # Include error in buffer
                print() # Newline after agent response

            except Exception as e:
                print(f"\n--- Fatal Error during agent step: {e} ---")
                break

    finally:
        # Clean up the temporary directory
        if temp_dir and temp_dir.exists():
            print(f"\nCleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        print("--- Interaction Finished ---")
