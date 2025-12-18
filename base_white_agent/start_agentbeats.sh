#!/bin/bash
# Start basic white agent for AgentBeats evaluation

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <domain_name>"
    echo "Example: $0 your-agent.ngrok.io"
    exit 1
fi

DOMAIN_NAME=$1

# Load OPENAI_API_KEY from .env file
if [ -f "../.env" ]; then
    export $(grep -v '^#' ../.env | xargs)
else
    echo "Error: .env file not found in project root"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not found in .env file"
    exit 1
fi

# Export environment variables for HTTPS
export HTTPS_ENABLED=true
export CLOUDRUN_HOST="${DOMAIN_NAME}"

echo "Starting basic white agent with AgentBeats..."
echo "HTTPS_ENABLED: ${HTTPS_ENABLED}"
echo "CLOUDRUN_HOST: ${CLOUDRUN_HOST}"

# Run agentbeats
agentbeats run_ctrl
