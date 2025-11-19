"""
Simple attempt storage - saves after every attempt.
Structure: store_path/model_id/question_id/attempts.json
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict


@dataclass
class Attempt:
    """Single attempt result"""
    attempt_id: int
    accuracy: float
    num_tokens: int
    num_turns: int
    timestamp: str
    metadata: Dict[str, Any] = None


class AttemptStore:
    """Stores attempts to disk organized by model/question"""
    
    def __init__(self, store_path: str = "./evaluation_results"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
    
    def save_attempt(self, model_id: str, question_id: str, attempt: Attempt) -> None:
        """Save attempt immediately to disk"""
        model_dir = self.store_path / model_id
        question_dir = model_dir / question_id
        question_dir.mkdir(parents=True, exist_ok=True)
        
        attempts_file = question_dir / "attempts.json"
        
        # Load existing attempts
        if attempts_file.exists():
            with open(attempts_file, 'r') as f:
                data = json.load(f)
        else:
            data = {"attempts": []}
        
        # Add new attempt
        data["attempts"].append(asdict(attempt))
        
        # Save
        with open(attempts_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_all(self) -> Dict[str, Dict[str, List[Attempt]]]:
        """Load all attempts: {model_id: {question_id: [Attempt, ...]}}"""
        results = {}
        
        if not self.store_path.exists():
            return results
        
        for model_dir in self.store_path.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_id = model_dir.name
            results[model_id] = {}
            
            for question_dir in model_dir.iterdir():
                if not question_dir.is_dir():
                    continue
                
                question_id = question_dir.name
                attempts_file = question_dir / "attempts.json"
                
                if attempts_file.exists():
                    with open(attempts_file, 'r') as f:
                        data = json.load(f)
                    
                    results[model_id][question_id] = [
                        Attempt(**a) for a in data["attempts"]
                    ]
        
        return results

