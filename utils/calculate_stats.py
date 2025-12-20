#!/usr/bin/env python3
"""
Calculate statistics from a JSONL results file.

Usage:
    python calculate_stats.py <jsonl_file_path>
"""

import json
import sys
from pathlib import Path

def calculate_stats(jsonl_path):
    """Calculate average statistics from a JSONL file."""
    with open(jsonl_path, 'r') as f:
        entries = [json.loads(line) for line in f]
    
    num_instances = len(entries)
    
    # Calculate averages
    avg_success_rate = sum(entry['success'] for entry in entries) / num_instances
    avg_num_tokens = sum(entry['num_tokens'] for entry in entries) / num_instances
    avg_accuracy = sum(entry['accuracy'] for entry in entries) / num_instances
    avg_num_turns = sum(entry['num_turns'] for entry in entries) / num_instances
    
    stats = {
        "num_instances": num_instances,
        "avg_success_rate": avg_success_rate,
        "avg_num_tokens": avg_num_tokens,
        "avg_accuracy": avg_accuracy,
        "avg_num_turns": avg_num_turns
    }
    
    # Save to JSON file in same directory
    output_path = Path(jsonl_path).parent / "stats.json"
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to {output_path}")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calculate_stats.py <jsonl_file_path>")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    calculate_stats(jsonl_path)
