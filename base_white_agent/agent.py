"""White agent implementation - the target agent being tested."""

import uvicorn
import dotenv
import logging
import os
import re
import uuid
from pathlib import Path
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities, Message, Part, Role, TextPart
from litellm import acompletion


# Load environment variables from .env file in project root
env_path = Path(__file__).parent.parent / ".env"
dotenv.load_dotenv(dotenv_path=env_path)

# Verify API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY not found in environment. "
        "Please ensure .env file exists in project root with OPENAI_API_KEY=your-key"
    )

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that can interact with a terminal.

Structure your responses as:

<reasoning>
Your thought process and analysis here.
</reasoning>

<answer>
Your response here (follow the format the user provides).
</answer>

"""


def prepare_white_agent_card(url):
    skill = AgentSkill(
        id="task_fulfillment",
        name="Task Fulfillment",
        description="Handles user requests and completes tasks using tools",
        tags=["general", "terminal", "tools"],
        examples=[],
    )
    card = AgentCard(
        name="terminal_bench_white_agent",
        description="White agent for solving terminal bench tasks",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class TerminalBenchWhiteAgentExecutor(AgentExecutor):
    """White agent executor using text-based JSON tool calling."""
    
    def __init__(self, model="gpt-5"):
        self.model = model
        self.ctx_id_to_messages = {}  # Maintain history per context_id
        self.ctx_id_to_tokens = {}  # Track token usage per context_id
        self.logger = logging.getLogger(__name__)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            # Get user input from context
            user_input = context.get_user_input()
            self.logger.debug(f"White agent received input for context {context.context_id}: {user_input[:100]}...")
            
            # Initialize or get message history for this context
            if context.context_id not in self.ctx_id_to_messages:
                self.ctx_id_to_messages[context.context_id] = [
                    {"role": "system", "content": SYSTEM_PROMPT}
                ]
                self.ctx_id_to_tokens[context.context_id] = 0
            
            messages = self.ctx_id_to_messages[context.context_id]
            
            # Add user input to message history
            messages.append({"role": "user", "content": user_input})
            
            # Call LLM asynchronously (no native tool calling - just text)
            api_params = {
                "model": self.model,
                "messages": messages,
            }
            
            if "gpt-5" not in self.model:
                api_params["temperature"] = 0.0
            
            self.logger.debug(f"Calling LLM with model={self.model}")
            response = await acompletion(**api_params)
            assistant_message = response.choices[0].message
            assistant_content = assistant_message.content or ""
            
            # Track token usage
            usage = response.usage if hasattr(response, 'usage') else None
            completion_tokens = 0
            cumulative_tokens = 0
            
            if usage and hasattr(usage, 'completion_tokens'):
                completion_tokens = usage.completion_tokens
                self.ctx_id_to_tokens[context.context_id] += completion_tokens
                cumulative_tokens = self.ctx_id_to_tokens[context.context_id]
                self.logger.debug(f"Tokens: {completion_tokens} completion, {cumulative_tokens} cumulative for context {context.context_id}")
            
            # Add assistant message to history
            messages.append({
                "role": "assistant",
                "content": assistant_content,
            })
            
            # Send response back with token metadata
            # Build Message manually to include metadata
            message = Message(
                role=Role.agent,
                parts=[Part(root=TextPart(text=assistant_content))],
                message_id=str(uuid.uuid4()),
                context_id=context.context_id,
                metadata={
                    "completion_tokens": completion_tokens,
                    "cumulative_tokens": cumulative_tokens
                }
            )
            await event_queue.enqueue_event(message)
            self.logger.debug(f"White agent sent response for context {context.context_id}")
            
        except Exception as e:
            # Log the full error details
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"White agent execute failed for context {context.context_id}: {e}")
            self.logger.error(f"Full traceback:\n{error_details}")
            # Re-raise so A2A SDK can handle it properly
            raise

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def start_white_agent(agent_name="terminal_bench_white_agent", host="localhost", port=9002, model="gpt-5"):
    print(f"Starting white agent with model={model}...")
    
    # Use public URL from environment if available (for AgentBeats/ngrok)
    base_url = os.getenv("AGENT_URL")
    card = prepare_white_agent_card(base_url)

    executor = TerminalBenchWhiteAgentExecutor(model=model)
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )
    
    # Build the Starlette app


    uvicorn.run(app.build(), host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="White Agent Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    args = parser.parse_args()
    
    start_white_agent(host=args.host, port=args.port)
