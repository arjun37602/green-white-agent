"""
JSONL-based results storage with caching support.
Structure: results_dir/model_name.jsonl (one line per task result)
"""

import json
import os
from pathlib import Path
from typing import Dict, Set, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import threading


@dataclass
class TaskResult:
    task_id: str
    attempt_id: int
    success: bool  # True if task completed successfully (all tests passed)
    num_tokens: int
    num_turns: int
    passed_test_cases: int
    total_test_cases: int
    accuracy: float
    timestamp: str
    execution_time: float
    message_history: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


class ResultsStore:
    """JSONL-based results storage with caching support"""
    
    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()  # Thread-safe writes
        
    def get_model_file(self, model_id: str) -> Path:
        """Get the JSONL file path for a model"""
        # Sanitize model_id for filename
        safe_model_id = model_id.replace('/', '_').replace('\\', '_')
        return self.results_dir / f"{safe_model_id}.jsonl"
    
    def load_completed_tasks(self, model_id: str) -> Set[str]:
        """Load set of completed task_ids for a model (for caching)"""
        model_file = self.get_model_file(model_id)
        completed = set()
        
        if not model_file.exists():
            return completed
        
        try:
            with open(model_file, 'r') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        completed.add(result['task_id'])
        except Exception as e:
            print(f"Warning: Error loading completed tasks for {model_id}: {e}")
        
        return completed
    
    def save_result(self, model_id: str, result: TaskResult) -> None:
        """Append a result to the model's JSONL file (thread-safe)"""
        model_file = self.get_model_file(model_id)
        
        with self._write_lock:
            with open(model_file, 'a') as f:
                json.dump(asdict(result), f)
                f.write('\n')
                f.flush() 
    
    def load_all_results(self, model_id: str) -> List[TaskResult]:
        """Load all results for a model"""
        model_file = self.get_model_file(model_id)
        results = []
        
        if not model_file.exists():
            return results
        
        try:
            with open(model_file, 'r') as f:
                for line in f:
                    if line.strip():
                        result_dict = json.loads(line)
                        results.append(TaskResult(**result_dict))
        except Exception as e:
            print(f"Warning: Error loading results for {model_id}: {e}")
        
        return results
    
    def get_summary_stats(self, model_id: str) -> Dict[str, Any]:
        """Get summary statistics for a model"""
        results = self.load_all_results(model_id)
        
        if not results:
            return {
                "model_id": model_id,
                "total_tasks": 0,
                "successful_tasks": 0,
                "success_rate": 0.0,
                "avg_accuracy": 0.0,
                "avg_tokens": 0.0,
                "avg_turns": 0.0,
                "total_execution_time": 0.0
            }
        
        successful_tasks = sum(1 for r in results if r.success)
        
        return {
            "model_id": model_id,
            "total_tasks": len(results),
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / len(results),
            "avg_accuracy": sum(r.accuracy for r in results) / len(results),
            "avg_tokens": sum(r.num_tokens for r in results) / len(results),
            "avg_turns": sum(r.num_turns for r in results) / len(results),
            "total_execution_time": sum(r.execution_time for r in results),
            "tasks_completed": list({r.task_id for r in results})
        }

