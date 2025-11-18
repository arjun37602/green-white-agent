#!/usr/bin/env python3
"""
White Agent Server for Terminal Bench - Tutorial Pattern

Simple wrapper to run the white agent as an A2A server compatible with the tutorial pattern.
"""

import argparse
import sys
from pathlib import Path

# Add green-white-agent to path
# File is now in: green-white-agent/agentbeats/scenarios/terminal_bench/agents/white_agent/white_agent_server.py
# green-white-agent root is at: parents[4]
GREEN_WHITE_AGENT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(GREEN_WHITE_AGENT_PATH))

from white_agent.agent import A2ATerminalBenchServer


def main():
    parser = argparse.ArgumentParser(description="Terminal Bench White Agent Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8341, help="Port to bind the server")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    args = parser.parse_args()
    
    # Create A2A server
    base_url = args.card_url or f"http://{args.host}:{args.port}"
    server = A2ATerminalBenchServer(port=args.port, host=args.host)
    
    # Update agent base_url if card-url provided
    if args.card_url:
        server.agent.base_url = args.card_url
        server.agent.agent_card.url = args.card_url
    
    # Update model if specified
    if args.model:
        server.agent.model = args.model
    
    # Run server
    import uvicorn
    uvicorn.run(server.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

