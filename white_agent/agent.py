"""White agent implementation - the target agent being tested."""

import uvicorn
import dotenv
import logging
import os
import json
import re
from pathlib import Path
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message
from litellm import completion


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
        self.ctx_id_to_tokens = {}  # Track cumulative tokens per context_id
        self.logger = logging.getLogger(__name__)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Get user input from context
        user_input = context.get_user_input()
        
        # Initialize or get message history for this context
        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = []
            self.ctx_id_to_tokens[context.context_id] = 0
        
        messages = self.ctx_id_to_messages[context.context_id]
        
        # Add user input to message history
        messages.append({"role": "user", "content": user_input})
        
        # Call LLM (no native tool calling - just text)
        api_params = {
            "model": self.model,
            "messages": messages,
        }
        
        if "gpt-5" not in self.model:
            api_params["temperature"] = 0.0
        
        response = completion(**api_params)
        assistant_message = response.choices[0].message
        assistant_content = assistant_message.content or ""
        
        usage = response.usage if hasattr(response, 'usage') else None
        if usage and hasattr(usage, 'completion_tokens'):
            tokens_this_turn = usage.total_tokens
            self.ctx_id_to_tokens[context.context_id] += tokens_this_turn
            self.logger.debug(f"Completion tokens : {usage.completion_tokens})")
        else:
            self.logger.warning(f"No completion tokens found in response")
        messages.append({
            "role": "assistant",
            "content": assistant_content,
            "tokens": usage.total_tokens if usage else 0
        })
        
        # Send response back
        await event_queue.enqueue_event(
            new_agent_text_message(
                assistant_content, context_id=context.context_id
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def start_white_agent(agent_name="terminal_bench_white_agent", host="localhost", port=9002):
    print("Starting white agent...")
    
    # Use public URL from environment if available (for AgentBeats/ngrok)
    base_url = os.getenv("AGENT_PUBLIC_URL", f"http://{host}:{port}")
    card = prepare_white_agent_card(base_url)

    request_handler = DefaultRequestHandler(
        agent_executor=TerminalBenchWhiteAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )
    
    # Build the Starlette app
    starlette_app = app.build()
    
    # Add endpoints for AgentBeats compatibility
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    
    async def get_agent_card_json(request):
        return JSONResponse(card.model_dump(mode='json', exclude_none=True))

    async def get_agent_status(request):
        return JSONResponse({
            "status": "healthy",
            "agent": agent_name,
            "version": "1.0.0"
        })

    # Add the agent card routes with and without .json extension
    starlette_app.routes.append(
        Route("/.well-known/agent-card.json", get_agent_card_json, methods=["GET"])
    )
    starlette_app.routes.append(
        Route("/.well-known/agent-card", get_agent_card_json, methods=["GET"])
    )
    starlette_app.routes.append(
        Route("/status", get_agent_status, methods=["GET"])
    )

    uvicorn.run(starlette_app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="White Agent Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    args = parser.parse_args()
    
    start_white_agent(host=args.host, port=args.port)
