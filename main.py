# code-agent/main.py

import typer
import pathlib
import os
import logging
from dotenv import load_dotenv

from agent import Agent, MessageType
from ui import (
    display_message,
    get_user_input,
    display_welcome,
    display_exit,
    display_error,
    display_status,
    console # Import the console object
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Typer app instance
app = typer.Typer(help="CLI AI agent to answer questions about code in a directory using Gemini.")

@app.command()
def main(
    root_dir: pathlib.Path = typer.Argument(
        ".",
        help="The root directory the agent should operate on.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True, # Ensure absolute path
    ),
    model_name: str = typer.Option(
        # "gemini-1.5-pro-latest", # Use 1.5 Pro for better function calling generally
        "gemini-1.5-flash-latest", # Or Flash for faster responses if sufficient
        "--model",
        "-m",
        help="The Gemini model to use (must support function calling)."
    )
):
    """
    Starts the interactive CLI agent.
    """
    logger.info(f"Starting Code Agent in directory: {root_dir}")
    logger.info(f"Using model: {model_name}")

    # Load API key from .env file
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        display_error("GOOGLE_API_KEY not found in environment variables or .env file.")
        logger.error("GOOGLE_API_KEY not found.")
        raise typer.Exit(code=1)

    # Initialize the agent
    try:
        code_agent = Agent(api_key=api_key, root_dir=root_dir, model_name=model_name)
    except Exception as e:
        display_error(f"Failed to initialize the agent: {e}")
        logger.exception("Agent initialization failed.")
        raise typer.Exit(code=1)

    # Display welcome message
    display_welcome(str(root_dir))

    # Main chat loop
    while True:
        try:
            user_input = get_user_input()

            if user_input.lower() in ["exit", "quit"]:
                display_exit()
                logger.info("User exited.")
                break

            if not user_input.strip():
                continue

            # Use Rich status for 'thinking' indicator
            with display_status("Agent is thinking...") as status:
                # Iterate through messages yielded by the agent
                for role, content in code_agent.send_message(user_input):
                    status.stop() # Stop status before printing message
                    if role == "user":
                        display_message(role, content)
                    elif role == "tool_call":
                        # Update status for tool execution
                        tool_name = content.split('(', 1)[0] # Extract tool name
                        status.update(f"Executing tool: {tool_name}...")
                        status.start() # Restart status spinner
                        display_message(role, content) # Show the call details
                    elif role == "tool_result":
                        status.stop() # Stop status before showing result
                        display_message(role, content)
                        # Update status as model processes result
                        status.update("Agent is processing tool result...")
                        status.start()
                    elif role == "agent":
                        status.stop() # Stop status for final answer
                        display_message(role, content, is_markdown=True)
                    else:
                        logger.warning(f"Unknown message role received: {role}")
                        display_message("error", f"Received unknown message type: {role}")

                    # If it was a tool call/result, the loop in agent continues,
                    # otherwise (agent final response), the inner loop breaks.
                    # The outer loop waits for next user input.


        except (KeyboardInterrupt, EOFError):
            display_exit()
            logger.info("User interrupted session.")
            break
        except Exception as e:
            display_error(f"An unexpected error occurred in the chat loop: {e}")
            logger.exception("Unexpected error in chat loop.")
            # Optionally decide whether to break or continue
            # break


if __name__ == "__main__":
    app()
