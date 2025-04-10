from dotenv import load_dotenv
from google.adk import Runner, Agent
from google.adk.sessions import InMemorySessionService
import asyncio
from google.genai import types # For creating message Content/Parts


APP_NAME = "ox"
USER_ID = "ox_user"
SESSION_ID = "ox_session"


def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city (e.g., "New York", "London", "Tokyo").

    Returns:
        dict: A dictionary containing the weather information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'report' key with weather details.
              If 'error', includes an 'error_message' key.
    """
    # Best Practice: Log tool execution for easier debugging
    print(f"--- Tool: get_weather called for city: {city} ---")
    city_normalized = city.lower().replace(" ", "") # Basic input normalization

    # Mock weather data for simplicity
    mock_weather_db = {
        "newyork": {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."},
        "london": {"status": "success", "report": "It's cloudy in London with a temperature of 15°C."},
        "tokyo": {"status": "success", "report": "Tokyo is experiencing light rain and a temperature of 18°C."},
    }

    # Best Practice: Handle potential errors gracefully within the tool
    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}


def make_runner() -> Runner:
  agent = Agent(
      name="weather_agent_v1",
      model="gemini-2.0-flash-exp",
      description="Provides weather information for specific cities.", # Crucial for delegation later
      instruction="You are a helpful weather assistant. Your primary goal is to provide current weather reports. "
                  "When the user asks for the weather in a specific city, "
                  "you MUST use the 'get_weather' tool to find the information. "
                  "Analyze the tool's response: if the status is 'error', inform the user politely about the error message. "
                  "If the status is 'success', present the weather 'report' clearly and concisely to the user. "
                  "Only use the tool when a city is mentioned for a weather request.",
      tools=[get_weather], # Make the tool available to this agent
  )

  session_service = InMemorySessionService()
  session = session_service.create_session(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID
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
  final_response_text = "Agent did not produce a final response." # Default

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
  await call_agent_async(r, "What is the weather like in London?")
  await call_agent_async(r, "How about Paris?")  # Expecting the tool's error message
  await call_agent_async(r, "Tell me the weather in New York")


if __name__ == "__main__":
  load_dotenv(override=True)
  asyncio.run(main())
