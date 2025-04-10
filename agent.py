from pathlib import Path

from dotenv import load_dotenv
from google.adk import Runner, Agent
from google.adk.sessions import InMemorySessionService
import asyncio
from google.genai import types

from context import OxContext
from tools import list_dir, read_file
import logging


APP_NAME = "ox"
USER_ID = "ox_user"
SESSION_ID = "ox_session"


def make_runner(root: Path) -> Runner:
  agent = Agent(
    name="weather_agent_v1",
    model="gemini-2.0-flash-exp",
    description="Provides code explanation",
    instruction="You are an expert software engineer with the goal of helping users navigate and understand the codebase. "
                "Use the tools available to you (list_dir, read_file) to analyze the codebase yourself to answer the user queries. "
                f"You're in `{root}` directory (root directory). You can work inside it using the tools. "
                "You can use paths relative to this directory when calling the tools. "
                "If the user is not specifying directory explicitly, assume they mean root.",
    tools=[list_dir, read_file],
  )

  session_service = InMemorySessionService()
  session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
    state={"ox_ctx": OxContext(root)}
  )

  runner = Runner(
    agent=agent,
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

  r = make_runner(Path("/Users/vorobevandrei/root/ox"))
  await call_agent_async(r, "What files are in this directory?")


if __name__ == "__main__":
  load_dotenv(override=True)
  asyncio.run(main())
