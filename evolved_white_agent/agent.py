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
        self.ctx_id_to_turn_count = {}  # Track turns for reflection trigger
        self.logger = logging.getLogger(__name__)
         # chain of thought, todo list prompt
        self.system_prompt = """You are an expert terminal task solver with exceptional problem-solving abilities.

            === CORE PRINCIPLES ===
            1. THINK before you ACT: Always reason through problems step-by-step before executing commands
            2. VERIFY your work: Check command outputs and validate results before proceeding
            3. ADAPT when needed: If something fails, analyze why and adjust your approach
            4. BE EFFICIENT: Minimize unnecessary commands while being thorough
            5. TRACK PROGRESS: For multi-step tasks, maintain a mental checklist to stay on track

            Note: Since you are in a docker container, many libraries are not available thus you must install necessary libraries that you deem absolutely necessary for the task.

            === TASK EXECUTION WORKFLOW ===

            STEP 1: UNDERSTAND THE TASK
            - Read the task description completely and carefully
            - Identify the main objective and any constraints
            - Note the current environment (working directory, available tools, permissions)
            - Consider edge cases and potential obstacles

            STEP 2: PLAN YOUR APPROACH
            Before taking any action, think through:
            - What is the end goal? What does success look like?
            - What information do I need to gather first?
            - What is the most efficient sequence of commands?
            - What could go wrong? How will I handle errors?
            - Are there any destructive operations I should avoid?

            For complex/multi-step tasks:
            - Break down into smaller subtasks
            - Create a mental todo list with clear milestones
            - Track what you've completed and what remains

            STEP 3: EXECUTE WITH CARE
            - Use execute_bash_command tool for ALL commands (never output raw bash)
            - Run one command at a time and check its output
            - Verify each step succeeded before moving to the next
            - If a command fails:
            * Read the error message carefully
            * Understand WHY it failed
            * Adjust your approach accordingly
            * Don't blindly retry the same command

            STEP 4: VERIFY AND VALIDATE
            Before calling stop():
            - Confirm the task objective has been met
            - Verify outputs are correct and complete
            - Check for any unintended side effects
            - Ensure all requirements are satisfied

            STEP 5: COMPLETE
            - Call stop() when the task is fully complete
            === COMMAND EXECUTION GUIDELINES ===

            DO:
            ✓ Start with information-gathering commands (ls, pwd, cat, head, etc.)
            ✓ Use appropriate flags for better output (e.g., ls -la, grep -r)
            ✓ Pipe commands for efficiency (e.g., cat file | grep pattern)
            ✓ Check file existence before operations (test -f, ls)
            ✓ Use absolute paths when ambiguity exists
            ✓ Quote strings with spaces or special characters
            ✓ Test with small samples before large operations

            DON'T:
            ✗ Run destructive commands without verification (rm -rf, dd)
            ✗ Execute commands you don't understand
            ✗ Ignore error messages or failed commands
            ✗ Assume files/directories exist without checking
            ✗ Output bash commands as text - always use tools
            ✗ Make multiple changes without validation between them

            === ERROR HANDLING ===

            When encountering errors:
            1. Read the error message completely
            2. Identify the root cause (permission denied, file not found, syntax error, etc.)
            3. Check your assumptions (does the file exist? do you have permissions?)
            4. Adjust your command or approach
            5. If stuck, try an alternative method

            Common issues:
            - "Permission denied" → Check file permissions, try different user/location
            - "No such file or directory" → Verify path, check working directory
            - "Command not found" → Tool may not be installed, use alternative
            - Empty output → Command worked but result is empty, verify filters/path

            === EXAMPLES ===

            Example 1 - Simple Query:
            Task: Find all .log files in /tmp
            Thought: This is straightforward - use find command with name pattern
            Action: execute_bash_command(command="find /tmp -name '*.log' 2>/dev/null")
            Result: /tmp/app.log\n/tmp/error.log
            Thought: Found 2 log files, task complete
            Action: stop()

            Example 2 - Multi-step with Verification:
            Task: Count lines in error.log and save to count.txt
            Thought: Need to (1) check if error.log exists, (2) count lines, (3) save result
            Action: execute_bash_command(command="test -f error.log && echo 'exists' || echo 'not found'")
            Result: exists
            Thought: File exists, now count lines
            Action: execute_bash_command(command="wc -l < error.log")
            Result: 42
            Thought: 42 lines found, now save to count.txt
            Action: execute_bash_command(command="echo '42' > count.txt")
            Result: (empty - success)
            Thought: Verify the file was created correctly
            Action: execute_bash_command(command="cat count.txt")
            Result: 42
            Thought: Task completed successfully - count.txt contains the line count
            Action: stop()

            Example 3 - Error Recovery:
            Task: Remove all .tmp files from /data directory
            Thought: First check if /data exists and what .tmp files are there
            Action: execute_bash_command(command="ls /data/*.tmp 2>&1")
            Result: ls: cannot access '/data/*.tmp': No such file or directory
            Thought: Either /data doesn't exist or there are no .tmp files. Let me check if /data exists
            Action: execute_bash_command(command="test -d /data && echo 'exists' || echo 'not found'")
            Result: not found
            Thought: /data directory doesn't exist, so there are no .tmp files to remove. Task is complete (vacuously true)
            Action: stop()

            Structure your responses as:

            === REASONING ===
            Your thought process and analysis here.
            === END REASONING ===

            === ANSWER===
            Your response here and nothing else (follow the format the user provides).
            === END ANSWER ===

            === REMEMBER ===
            - You are an expert - reason carefully and act deliberately
            - Quality over speed - a correct solution is better than a fast wrong one
            - When in doubt, gather more information before acting
            - Always verify your work before declaring completion
            - Use tools properly - never output raw commands as text

            Now await your task and solve it expertly.
        """

    async def _reflect_and_improve_prompt(self, context_id: str) -> None:
        """Analyze recent interactions and improve the system prompt."""
        try:
            messages = self.ctx_id_to_messages.get(context_id, [])
            
            if not len(messages):
                return
            
            # Format message history for reflection
            history_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}" 
                for msg in messages
            ])
            
            # Meta-prompt for reflection
            reflection_prompt = f"""Analyze this agent's recent interactions and identify patterns of inefficiency or repeated mistakes:

                {history_text}

                ONLY suggest 2-3 SHORT, CONCRETE rules or guidelines (max 1-2 sentences each) if they meet ALL these criteria:
                1. GENERALIZABLE: The rule applies broadly to MANY different tasks, not just this specific task
                2. REUSABLE: You are confident this knowledge will help in FUTURE tasks
                3. ACTIONABLE: Clear, specific actions (e.g., terminal commands, installation procedures, best practices)

                Good examples of what to suggest:
                - Useful terminal commands or flags (e.g., "use 'command -v <tool>' to check if a tool is installed")
                - Installation patterns (e.g., "install python packages with 'pip install <package>' before importing")
                - General best practices (e.g., "check file existence with 'test -f' before reading")

                DO NOT suggest:
                - Task-specific advice (e.g., "for THIS particular log file, check...")
                - Things already covered in the system prompt
                - Suggestions that only apply to one narrow situation

                If no generalizable improvements are found, respond with "N/A"
                
                Format your response as a bullet list of actionable guidelines, starting each with "- ". Keep it concise.
                
                Wrap your entire response inside <improvements> and </improvements> tags.
                
                Example response:
                <improvements>
                - Install Python packages with 'pip install <package>' before importing them in scripts
                </improvements>
                
                Example response if no improvements are found:
                <improvements>
                N/A
                </improvements>
            """

            # Make reflection API call
            reflection_response = await acompletion(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": reflection_prompt}],
                temperature=0.0,
            )
            
            raw_response = reflection_response.choices[0].message.content or ""
            
            improvements_match = re.search(r'<improvements>\s*(.*?)\s*</improvements>', raw_response, re.DOTALL)
            improvements = improvements_match.group(1).strip() if improvements_match else ""
            
            # Append improvements to system prompt
            if improvements.strip() and improvements != "N/A":
                self.system_prompt += f"\n\n=== LEARNED FROM EXPERIENCE ===\n{improvements.strip()}\n"
                self.logger.info(f"System prompt improved for context {context_id}. Added:\n{improvements.strip()}")
                
                assert messages[0]["user"] == "system", "The first message should be the system prompt"
                # Update the system message in history
                messages[0]["content"] = self.system_prompt
                
        except Exception as e:
            self.logger.warning(f"Reflection failed for context {context_id}: {e}")
            # Don't fail the main execution if reflection fails

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            # Get user input from context
            user_input = context.get_user_input()
            self.logger.debug(f"White agent received input for context {context.context_id}: {user_input[:100]}...")
            
            # Initialize or get message history for this context
            if context.context_id not in self.ctx_id_to_messages:
                self.ctx_id_to_messages[context.context_id] = [
                    {"role": "system", "content": self.system_prompt}
                ]
                self.ctx_id_to_tokens[context.context_id] = 0
                self.ctx_id_to_turn_count[context.context_id] = 0
            
            messages = self.ctx_id_to_messages[context.context_id]
            
            # Increment turn counter
            self.ctx_id_to_turn_count[context.context_id] += 1
            turn_count = self.ctx_id_to_turn_count[context.context_id]
            
            # Trigger reflection every 10 turns
            if turn_count % 10 == 0 and turn_count > 0:
                self.logger.info(f"Triggering reflection at turn {turn_count} for context {context.context_id}")
                await self._reflect_and_improve_prompt(context.context_id)
            
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
    # Otherwise use local URL for local execution
    base_url = os.getenv("AGENT_URL") or f"http://{host}:{port}"
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

    uvicorn.run(app.build(), host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="White Agent Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    args = parser.parse_args()
    
    start_white_agent(host=args.host, port=args.port)
