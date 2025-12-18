#!/usr/bin/env python3
"""Analyze Terminal Bench results from JSONL file."""

import json
import sys
from pathlib import Path


def analyze_results(jsonl_file, output_file=None):
    """Calculate average metrics from JSON results file."""
    
    # Read all results (handle both JSONL and concatenated JSON objects)
    results = []
    with open(jsonl_file, 'r') as f:
        content = f.read()
    
    # Split on }{ pattern to separate concatenated JSON objects
    json_objects = content.replace('}\n{', '}\n===SPLIT===\n{').split('===SPLIT===')
    
    for obj_str in json_objects:
        obj_str = obj_str.strip()
        if obj_str:
            try:
                results.append(json.loads(obj_str))
            except json.JSONDecodeError:
                # Skip invalid JSON
                continue
    
    if not results:
        print("No results found in file")
        return
    
    # Calculate averages
    total = len(results)
    avg_accuracy = sum(r['accuracy'] for r in results) / total
    avg_score = sum(1 if r['success'] else 0 for r in results) / total
    avg_tokens = sum(r['num_tokens'] for r in results) / total
    avg_time = sum(r['execution_time'] for r in results) / total
    avg_turns = sum(r['num_turns'] for r in results) / total
    
    # Format output
    output = f"""Terminal Bench Results Analysis
{'='*50}
Total Tasks: {total}

Average Metrics:
  Score (Pass Rate):  {avg_score:.2%} ({sum(1 if r['success'] else 0 for r in results)}/{total} passed)
  Accuracy:           {avg_accuracy:.4f}
  Tokens per Task:    {avg_tokens:.1f}
  Time per Task:      {avg_time:.2f}s
  Turns per Task:     {avg_turns:.1f}
"""
    
    # Print to terminal
    print(output)
    
    # Write to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output)
        print(f"Results written to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <jsonl_file> [output_file]")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(jsonl_file).exists():
        print(f"Error: File not found: {jsonl_file}")
        sys.exit(1)
    
    analyze_results(jsonl_file, output_file)

