#!/bin/bash
# Start green agent for AgentBeats evaluation

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <domain_name>"
    echo "Example: $0 your-green-agent.ngrok.io"
    exit 1
fi

DOMAIN_NAME=$1

# Export environment variables for HTTPS (no OPENAI_API_KEY needed for green agent)
export HTTPS_ENABLED=true
export CLOUDRUN_HOST="${DOMAIN_NAME}"

echo "Starting green agent with AgentBeats..."
echo "HTTPS_ENABLED: ${HTTPS_ENABLED}"
echo "CLOUDRUN_HOST: ${CLOUDRUN_HOST}"

# Run agentbeats
agentbeats run_ctrl
