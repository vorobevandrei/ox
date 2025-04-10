from pathlib import Path

from dotenv import load_dotenv
from google.adk import Runner, Agent
from google.adk.sessions import InMemorySessionService
import asyncio
from google.genai import types

from context import OxContext
from tools import list_dir

APP_NAME = "ox"
USER_ID = "ox_user"
SESSION_ID = "ox_session"


def make_runner() -> Runner:
  agent = Agent(
    name="weather_agent_v1",
    model="gemini-2.0-flash-exp",
    description="Provides code explanation",
    instruction="You are a helpful weather assistant. Your primary goal is to provide current weather reports. "
                "When the user asks for the weather in a specific city, "
                "you MUST use the 'get_weather' tool to find the information. "
                "Analyze the tool's response: if the status is 'error', inform the user politely about the error message. "
                "If the status is 'success', present the weather 'report' clearly and concisely to the user. "
                "Only use the tool when a city is mentioned for a weather request.",
    tools=[list_dir],
  )

  session_service = InMemorySessionService()
  session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
    state={"ox_ctx": OxContext(root=Path("/Users/vorobevandrei/root/ox"))}
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
  r = make_runner()
  await call_agent_async(r, "What files are in this directory?")


if __name__ == "__main__":
  load_dotenv(override=True)
  asyncio.run(main())
