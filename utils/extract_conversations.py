#!/usr/bin/env python3
"""
Extract conversations from JSONL results file.

Usage:
    python extract_conversations.py <jsonl_file_path>
"""

import json
import sys
from pathlib import Path

def extract_conversations(jsonl_path):
    """Extract task_id -> message_history mapping from JSONL file."""
    conversations = {}
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            task_id = entry['task_id']
            message_history = entry['message_history']
            conversations[task_id] = message_history
    
    # Save to conversations.json in same directory
    output_path = Path(jsonl_path).parent / "conversations.json"
    with open(output_path, 'w') as f:
        json.dump(conversations, f, indent=2)
    
    print(f"Extracted {len(conversations)} conversations")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_conversations.py <jsonl_file_path>")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    extract_conversations(jsonl_path)
