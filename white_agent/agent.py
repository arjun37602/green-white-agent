"""
Basic A2A Compatible Agent using Google ADK
Install: pip install google-adk[a2a]
"""
from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create a basic agent
agent = Agent(
    model="gemini-2.0-flash-exp",
    name="white_agent",
    description="A versatile agent capable of handling various tasks and problem-solving.",
    instruction="You are a helpful AI assistant. Respond to user queries clearly and concisely."
)

# Make it A2A compatible - this creates an ASGI app
a2a_app = to_a2a(
    agent,
    port=8001,  # Port where this agent will be served
)

# To run this agent as an A2A server:
# 1. Save this file as agent.py
# 2. Run: uvicorn agent:a2a_app --host 0.0.0.0 --port 8001
#
# OR use the ADK CLI:
# adk api_server --a2a --port 8001 path/to/agent.py

if __name__ == "__main__":
    import uvicorn
    print("Starting A2A agent server on port 8001...")
    print("Agent card will be available at: http://localhost:8001/.well-known/agent.json")
    uvicorn.run(a2a_app, host="0.0.0.0", port=8001)