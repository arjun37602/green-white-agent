#!/bin/bash
# Simple setup script using conda and pip
set -e

ENV_NAME="green-white-agent"

echo "Creating conda environment with Python 3.12..."
# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing conda environment: ${ENV_NAME}"
    conda env remove -n "${ENV_NAME}" -y
fi

# Create new conda environment
conda create -n "${ENV_NAME}" python=3.12 -y

echo ""
echo "Activating conda environment..."
# Initialize conda for this script (works for both bash and zsh)
if [ -n "$ZSH_VERSION" ]; then
    eval "$(conda shell.zsh hook)"
elif [ -n "$BASH_VERSION" ]; then
    eval "$(conda shell.bash hook)"
else
    # Fallback: source conda.sh directly
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "${ENV_NAME}"

echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

echo ""
echo "Installing green-white-agent and dependencies..."
pip install -e .

echo ""
echo "Verifying installation..."
if python -c "import terminal_bench" 2>/dev/null; then
    echo "✓ terminal-bench is installed"
else
    echo "⚠️  Warning: terminal-bench import test failed (expected due to broken imports)"
fi

if python -c "import sys; import os; sp = os.path.join(sys.prefix, 'lib', 'python3.12', 'site-packages', 'terminal_bench'); print('Found' if os.path.exists(sp) else 'Not found')" 2>/dev/null | grep -q "Found"; then
    echo "✓ terminal-bench found in site-packages"
else
    echo "⚠️  Could not verify terminal-bench location (but installation may have succeeded)"
fi


