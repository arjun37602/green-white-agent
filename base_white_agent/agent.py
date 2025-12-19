"""White agent implementation - the target agent being tested."""

import uvicorn
import dotenv
import logging
import os
import re
import sys
import uuid
import traceback
import asyncio
import signal
from pathlib import Path
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities, Message, Part, Role, TextPart
from litellm import acompletion
import litellm


# Load environment variables from .env file in project root
env_path = Path(__file__).parent.parent / ".env"
dotenv.load_dotenv(dotenv_path=env_path)

# Verify API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY not found in environment. "
        "Please ensure .env file exists in project root with OPENAI_API_KEY=your-key"
    )

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configure LiteLLM logging for debugging
litellm.set_verbose = False  # Set to True for full LiteLLM debug output

# Track active requests for debugging
_active_requests = {}
_request_counter = 0
_request_lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None


def _get_request_lock():
    """Lazy initialization of request lock for async compatibility."""
    global _request_lock
    if _request_lock is None:
        _request_lock = asyncio.Lock()
    return _request_lock


# Global exception handler for uncaught asyncio exceptions
def handle_exception(loop, context):
    """Handle uncaught exceptions in asyncio event loop."""
    msg = context.get("exception", context["message"])
    logger.critical(f"🔴 ASYNCIO UNCAUGHT EXCEPTION: {msg}")
    logger.critical(f"🔴 Context: {context}")
    if "exception" in context:
        logger.critical(f"🔴 Full traceback:\n{traceback.format_exception(type(context['exception']), context['exception'], context['exception'].__traceback__)}")

SYSTEM_PROMPT = """You are a helpful assistant that can interact with a terminal.

Structure your responses as:

=== REASONING ===
Your thought process and analysis here.
=== END REASONING ===

=== ANSWER===
Your response here and nothing else (follow the format the user provides).
=== END ANSWER ===

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
        self._total_requests = 0
        self._active_requests = 0
        self._failed_requests = 0
        self._successful_requests = 0

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        global _request_counter
        
        # Track this request
        request_id = None
        start_time = None
        
        try:
            import time
            start_time = time.time()
            
            async with _get_request_lock():
                _request_counter += 1
                request_id = _request_counter
                self._total_requests += 1
                self._active_requests += 1
                _active_requests[request_id] = {
                    "context_id": context.context_id,
                    "start_time": start_time,
                    "status": "started"
                }
            
            self.logger.info(f"🔵 [REQ-{request_id}] NEW REQUEST - context={context.context_id}, active={self._active_requests}, total={self._total_requests}")
            
            user_input = context.get_user_input()
            input_preview = user_input[:200].replace('\n', '\\n') if user_input else "(empty)"
            self.logger.debug(f"🔵 [REQ-{request_id}] Input preview: {input_preview}...")
            
            if context.context_id not in self.ctx_id_to_messages:
                self.ctx_id_to_messages[context.context_id] = [
                    {"role": "system", "content": SYSTEM_PROMPT}
                ]
                self.ctx_id_to_tokens[context.context_id] = 0
                self.logger.info(f"🔵 [REQ-{request_id}] New context initialized: {context.context_id}")
            
            messages = self.ctx_id_to_messages[context.context_id]
            messages.append({"role": "user", "content": user_input})
            
            # Log memory usage
            try:
                import resource
                mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
                self.logger.debug(f"🔵 [REQ-{request_id}] Memory usage: {mem_usage:.1f} MB")
            except Exception:
                pass
            
            api_params = {
                "model": self.model,
                "messages": messages,
            }
            
            if "gpt-5" not in self.model:
                api_params["temperature"] = 0.0
            
            max_api_tries = 3  # Increased retries
            last_api_error = None
            response = None
            
            for api_attempt in range(max_api_tries):
                try:
                    self.logger.info(f"🔵 [REQ-{request_id}] Calling LLM with model={self.model} (attempt {api_attempt + 1}/{max_api_tries})")
                    _active_requests[request_id]["status"] = f"llm_call_attempt_{api_attempt + 1}"
                    
                    response = await acompletion(**api_params)
                    
                    self.logger.info(f"🟢 [REQ-{request_id}] LLM call successful (attempt {api_attempt + 1})")
                    break
                    
                except asyncio.CancelledError:
                    self.logger.warning(f"🟠 [REQ-{request_id}] Request cancelled during LLM call")
                    raise
                    
                except Exception as api_error:
                    last_api_error = api_error
                    error_type = type(api_error).__name__
                    error_msg = str(api_error)
                    
                    # Log detailed error info
                    self.logger.error(f"🔴 [REQ-{request_id}] LLM API error (attempt {api_attempt + 1}/{max_api_tries})")
                    self.logger.error(f"🔴 [REQ-{request_id}] Error type: {error_type}")
                    self.logger.error(f"🔴 [REQ-{request_id}] Error message: {error_msg}")
                    self.logger.error(f"🔴 [REQ-{request_id}] Traceback:\n{traceback.format_exc()}")
                    
                    # Check for specific error types
                    if "rate_limit" in error_msg.lower() or "429" in error_msg:
                        self.logger.warning(f"🟠 [REQ-{request_id}] Rate limit detected, waiting 5s before retry...")
                        await asyncio.sleep(5)
                    elif "timeout" in error_msg.lower():
                        self.logger.warning(f"🟠 [REQ-{request_id}] Timeout detected, waiting 2s before retry...")
                        await asyncio.sleep(2)
                    elif "connection" in error_msg.lower():
                        self.logger.warning(f"🟠 [REQ-{request_id}] Connection error, waiting 2s before retry...")
                        await asyncio.sleep(2)
                    elif api_attempt < max_api_tries - 1:
                        self.logger.warning(f"🟠 [REQ-{request_id}] Unknown error, waiting 2s before retry...")
                        await asyncio.sleep(2)
                    else:
                        self.logger.error(f"🔴 [REQ-{request_id}] LLM API call failed after {max_api_tries} attempts")
                        raise last_api_error
            
            if response is None:
                self.logger.error(f"🔴 [REQ-{request_id}] No response received after {max_api_tries} attempts")
                raise RuntimeError(f"Failed to get LLM response after {max_api_tries} attempts")
            
            _active_requests[request_id]["status"] = "processing_response"
            
            assistant_message = response.choices[0].message
            assistant_content = assistant_message.content or ""
            
            usage = response.usage if hasattr(response, 'usage') else None
            completion_tokens = 0
            cumulative_tokens = 0
            
            if usage and hasattr(usage, 'completion_tokens'):
                completion_tokens = usage.completion_tokens
                self.ctx_id_to_tokens[context.context_id] += completion_tokens
                cumulative_tokens = self.ctx_id_to_tokens[context.context_id]
                self.logger.debug(f"🔵 [REQ-{request_id}] Tokens: {completion_tokens} completion, {cumulative_tokens} cumulative")
            
            messages.append({
                "role": "assistant",
                "content": assistant_content,
            })
            
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
            
            _active_requests[request_id]["status"] = "enqueueing_response"
            await event_queue.enqueue_event(message)
            
            elapsed = time.time() - start_time
            self._successful_requests += 1
            self.logger.info(f"🟢 [REQ-{request_id}] COMPLETED in {elapsed:.2f}s - context={context.context_id}, tokens={cumulative_tokens}")
            
        except asyncio.CancelledError:
            self.logger.warning(f"🟠 [REQ-{request_id}] Request was cancelled")
            self._failed_requests += 1
            raise
            
        except Exception as e:
            error_type = type(e).__name__
            elapsed = time.time() - start_time if start_time else 0
            self._failed_requests += 1
            
            self.logger.error(f"🔴 [REQ-{request_id}] FAILED after {elapsed:.2f}s - context={context.context_id}")
            self.logger.error(f"🔴 [REQ-{request_id}] Error type: {error_type}")
            self.logger.error(f"🔴 [REQ-{request_id}] Error message: {str(e)}")
            self.logger.error(f"🔴 [REQ-{request_id}] Full traceback:\n{traceback.format_exc()}")
            self.logger.error(f"🔴 [REQ-{request_id}] Stats: success={self._successful_requests}, failed={self._failed_requests}, active={self._active_requests}")
            raise
            
        finally:
            # Always clean up
            async with _get_request_lock():
                self._active_requests -= 1
                if request_id in _active_requests:
                    del _active_requests[request_id]
            
            self.logger.debug(f"🔵 [REQ-{request_id}] Cleanup complete, active requests now: {self._active_requests}")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        self.logger.warning(f"🟠 Cancel requested for context: {context.context_id}")
        raise NotImplementedError


def start_white_agent(agent_name="terminal_bench_white_agent", host="localhost", port=9002, model="gpt-5"):
    logger.info(f"🚀 Starting white agent with model={model} on {host}:{port}")
    logger.info(f"🚀 Python version: {sys.version}")
    logger.info(f"🚀 Process ID: {os.getpid()}")
    
    # Log memory at startup
    try:
        import resource
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
        logger.info(f"🚀 Initial memory usage: {mem_usage:.1f} MB")
    except Exception as e:
        logger.debug(f"Could not get memory usage: {e}")
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.warning(f"🟠 Received signal {sig_name} ({signum})")
        logger.warning(f"🟠 Active requests at shutdown: {_active_requests}")
        logger.warning(f"🟠 Shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    base_url = os.getenv("AGENT_URL") or f"http://{host}:{port}"
    card = prepare_white_agent_card(base_url)

    executor = TerminalBenchWhiteAgentExecutor(model=model)
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    # Create the app with error handling
    try:
        app = A2AStarletteApplication(
            agent_card=card,
            http_handler=request_handler,
        )
        logger.info(f"🚀 A2A application created successfully")
    except Exception as e:
        logger.critical(f"🔴 Failed to create A2A application: {e}")
        logger.critical(f"🔴 Traceback:\n{traceback.format_exc()}")
        raise

    # Configure uvicorn with error callbacks
    config = uvicorn.Config(
        app.build(),
        host=host,
        port=port,
        backlog=10000,  # Increase backlog for high-parallelism scenarios
        limit_max_requests=100000,  # Allow many requests before restart
        timeout_keep_alive=300,  # Keep connections alive for 5min (for long LLM calls)
        h11_max_incomplete_event_size=100 * 1024 * 1024,  # 100MB for large responses
        log_level="info",
        access_log=True,
        # Note: NO limit_concurrency - we want to handle all concurrent requests, not reject them
    )
    
    server = uvicorn.Server(config)
    
    logger.info(f"🚀 Starting uvicorn server...")
    logger.info(f"🚀 Configuration: backlog=10000, timeout_keep_alive=300, limit_max_requests=100000")
    
    try:
        server.run()
    except Exception as e:
        logger.critical(f"🔴 Uvicorn server crashed: {e}")
        logger.critical(f"🔴 Traceback:\n{traceback.format_exc()}")
        logger.critical(f"🔴 Active requests at crash: {_active_requests}")
        raise
    finally:
        logger.warning(f"🟠 White agent shutting down")
        logger.warning(f"🟠 Final stats: executor.total={executor._total_requests}, success={executor._successful_requests}, failed={executor._failed_requests}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="White Agent Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    args = parser.parse_args()
    
    start_white_agent(host=args.host, port=args.port)
