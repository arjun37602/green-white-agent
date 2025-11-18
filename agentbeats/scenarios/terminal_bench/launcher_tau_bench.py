"""Terminal Bench Launcher - Tau-Bench Pattern

Launches green and white agents and sends evaluation task.
"""

import multiprocessing
import json
import asyncio
from pathlib import Path
import sys

# Add paths
# File is now in: green-white-agent/agentbeats/scenarios/terminal_bench/launcher_tau_bench.py
SCENARIO_DIR = Path(__file__).resolve().parent
GREEN_WHITE_AGENT_PATH = SCENARIO_DIR.parents[3]  # Go up to green-white-agent root
sys.path.insert(0, str(GREEN_WHITE_AGENT_PATH))
sys.path.insert(0, str(SCENARIO_DIR / "agents" / "green_agent"))
sys.path.insert(0, str(SCENARIO_DIR / "agents" / "white_agent"))

from green_agent_tau_bench import start_green_agent, send_message_to_agent
import white_agent_server


async def wait_agent_ready(url: str, timeout: int = 10) -> bool:
    """Wait until the A2A server is ready."""
    import httpx
    from a2a.client import A2ACardResolver
    
    retry_cnt = 0
    while retry_cnt < timeout:
        retry_cnt += 1
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                resolver = A2ACardResolver(httpx_client=client, base_url=url)
                card = await resolver.get_agent_card()
                if card is not None:
                    return True
        except Exception:
            pass
        print(f"Agent not ready yet, retrying {retry_cnt}/{timeout}...")
        await asyncio.sleep(1)
    return False


async def launch_evaluation():
    """Launch complete Terminal Bench evaluation."""
    # Start green agent
    print("Launching green agent...")
    green_address = ("localhost", 8340)
    green_url = f"http://{green_address[0]}:{green_address[1]}"
    p_green = multiprocessing.Process(
        target=start_green_agent, args=(green_address[0], green_address[1])
    )
    p_green.start()
    
    if not await wait_agent_ready(green_url):
        print("Error: Green agent not ready in time")
        p_green.terminate()
        return
    print("Green agent is ready.")
    
    # Start white agent
    print("Launching white agent...")
    white_address = ("localhost", 8341)
    white_url = f"http://{white_address[0]}:{white_address[1]}"
    
    def run_white_agent():
        import sys
        sys.argv = ["white_agent_server.py", "--host", white_address[0], "--port", str(white_address[1])]
        white_agent_server.main()
    
    p_white = multiprocessing.Process(target=run_white_agent)
    p_white.start()
    
    if not await wait_agent_ready(white_url):
        print("Error: White agent not ready in time")
        p_green.terminate()
        p_white.terminate()
        return
    print("White agent is ready.")
    
    # Send task description to green agent
    print("Sending task description to green agent...")
    task_config = {
        "limit": 3,  # Evaluate 3 tasks
        # "task_ids": [1, 2, 3],  # Optional: specific task IDs
    }
    task_text = f"""
Your task is to evaluate the Terminal Bench agent located at:
<white_agent_url>
{white_url}
</white_agent_url>
You should use the following task configuration:
<task_config>
{json.dumps(task_config, indent=2)}
</task_config>
"""
    print("Task description:")
    print(task_text)
    print("Sending...")
    
    try:
        response = await send_message_to_agent(green_url, task_text)
        print("Response from green agent:")
        # Extract text from response
        from a2a.utils import get_text_parts
        from a2a.types import Message
        
        # Handle different response types
        if hasattr(response, 'root') and hasattr(response.root, 'result'):
            result = response.root.result
            if isinstance(result, Message):
                text_parts = get_text_parts(result.parts)
                for text in text_parts:
                    print(text)
            else:
                print(f"Response: {result}")
        else:
            print(f"Response: {response}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nEvaluation complete. Terminating agents...")
    p_green.terminate()
    p_green.join()
    p_white.terminate()
    p_white.join()
    print("Agents terminated.")


if __name__ == "__main__":
    asyncio.run(launch_evaluation())

