import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional, Dict

from dotenv import load_dotenv
from google.adk import Runner, Agent
from google.adk.sessions import InMemorySessionService
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from context import OxContext, CTX_KEY
from tools import ToolBox

load_dotenv(override=True)
WORK_DIR = Path(os.environ["WORK_DIR"]).resolve()


def before_tool_callback(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> Optional[Dict]:
  tool_context.state[CTX_KEY] = OxContext(WORK_DIR)


root_agent = Agent(
  name="ox",
  model="gemini-2.0-flash-exp",
  description="Provides code explanation",
  instruction="You are an expert software engineer with the goal of helping users navigate and understand the codebase. "
              f"Use the tools available to you ({", ".join(ToolBox.tools_names())}) to analyze the codebase yourself to answer the user queries. "
              f"You're in `{WORK_DIR}` directory (root directory). You can work inside it using the tools. Refer to the root as `.`"
              "You can use paths relative to this directory when calling the tools. "
              "If the user is not specifying directory explicitly, assume they mean root.",
  tools=ToolBox.tools,
  before_tool_callback=before_tool_callback
)


# the rest is for running without adk


APP_NAME = "ox"
USER_ID = "ox_user"
SESSION_ID = "ox_session"


def make_runner(root: Path) -> Runner:
  session_service = InMemorySessionService()
  session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
    state={CTX_KEY: OxContext(root)}
  )

  runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service
  )

  return runner


async def call_agent_async(runner: Runner, query: str):
  print(f"\n>>> User Query: {query}")

  content = types.Content(role='user', parts=[types.Part(text=query)])
  final_response_text = "Agent did not produce a final response."  # Default

  async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
    if event.is_final_response():
      if event.content and event.content.parts:
        final_response_text = event.content.parts[0].text
      elif event.actions and event.actions.escalate:
        final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
      break

  print(f"<<< Agent Response: {final_response_text}")


async def main():
  # suppress annoying warning
  logging.getLogger("google_genai.types").setLevel(logging.ERROR)
  runner = make_runner(Path("/Users/vorobevandrei/root/code-migration"))
  await call_agent_async(runner, "What files are in this directory?")


if __name__ == "__main__":
  load_dotenv(override=True)
  asyncio.run(main())
