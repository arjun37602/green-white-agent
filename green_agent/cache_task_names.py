#!/usr/bin/env python3
"""
Utility to cache Terminal-Bench task names from the dataset.
Run this once to generate a cached list of task IDs.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock problematic imports
sys.modules['terminal_bench.harness'] = MagicMock()
sys.modules['terminal_bench.harness.harness'] = MagicMock()
sys.modules['terminal_bench.harness.models'] = MagicMock()
sys.modules['terminal_bench.agents'] = MagicMock()
sys.modules['terminal_bench.agents.base_agent'] = MagicMock()
sys.modules['terminal_bench.db'] = MagicMock()

from terminal_bench.dataset.dataset import Dataset


def cache_task_names(version="0.1.1", output_file="task_names_cache.json"):
    """
    Cache task names from Terminal-Bench dataset.
    Auto-downloads dataset if not present.
    
    Args:
        version: Dataset version (default: "0.1.1")
        output_file: Output JSON file path
    """
    # Get dataset path using Dataset object
    dataset_instance = Dataset.__new__(Dataset)
    dataset_path = dataset_instance._get_cache_path("terminal-bench-core", version)
    
    if not dataset_path.exists():
        print(f"📥 Dataset not found at: {dataset_path}")
        print(f"⏳ Auto-downloading Terminal-Bench dataset version {version}...")
        print(f"   This may take a few minutes on first run...")
        
        # Initialize Dataset with name and version - will trigger download
        try:
            dataset = Dataset(name="terminal-bench-core", version=version)
            dataset_path = dataset.config.path
            print(f"✅ Downloaded dataset to: {dataset_path}")
        except Exception as e:
            print(f"❌ Failed to download dataset: {e}")
            print(f"\nTry manually downloading with:")
            print(f"  python -c 'from terminal_bench.dataset.dataset import Dataset; Dataset(name=\"terminal-bench-core\", version=\"{version}\")'")
            return None
    else:
        print(f"📦 Loading dataset from: {dataset_path}")
        dataset = Dataset(path=dataset_path)
    
    # Extract task names
    task_names = []
    for task_path in dataset:
        task_names.append(task_path.name)
    
    print(f"✅ Found {len(task_names)} tasks")
    
    # Save to cache file
    cache_data = {
        "version": version,
        "dataset_path": str(dataset_path),
        "task_count": len(task_names),
        "task_ids": task_names
    }
    
    output_path = Path(__file__).parent / output_file
    with open(output_path, "w") as f:
        json.dump(cache_data, f, indent=2)
    
    print(f"💾 Cached {len(task_names)} task names to: {output_path}")
    print(f"\nFirst 10 tasks: {task_names[:10]}")
    
    return cache_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cache Terminal-Bench task names")
    parser.add_argument("--version", default="0.1.1", help="Dataset version (default: 0.1.1)")
    parser.add_argument("--output", default="task_names_cache.json", help="Output file (default: task_names_cache.json)")
    
    args = parser.parse_args()
    
    cache_task_names(version=args.version, output_file=args.output)

