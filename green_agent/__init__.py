"""
Green Agent Package

This package contains the green agent components for Terminal Bench evaluation.
The green agent is responsible for loading tasks and evaluating white agent responses.
"""

from .terminal_bench_runner import GreenAgentTerminalBench

__all__ = ['GreenAgentTerminalBench']
