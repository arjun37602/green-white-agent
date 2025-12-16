#!/bin/bash

# Get the project root directory (parent of white_agent/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Load .env file if it exists (for local development)
if [ -f "${PROJECT_ROOT}/.env" ]; then
    echo "Loading environment from .env file..."
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

# Verify OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set. Please set it in .env file or export it."
    exit 1
fi

# Add project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Change to project root before running
cd "${PROJECT_ROOT}"

python -c "from white_agent.agent import start_white_agent; start_white_agent(host='${HOST:-0.0.0.0}', port=${AGENT_PORT:-8001})"