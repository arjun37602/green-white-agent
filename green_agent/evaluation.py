"""
Minimal evaluation system with handler registry.
Handlers define how to compare models (get_label), train (loss), and report (leaderboard).
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Callable, Any
from collections import defaultdict
from dataclasses import dataclass

from .attempt_store import Attempt


@dataclass
class Battle:
    """A single comparison between two models"""
    model_a: str
    model_b: str
    question_id: str
    attempt_a: Attempt
    attempt_b: Attempt
    label: str  # "a_wins", "b_wins", "tie", "both"


class Handler:
    """Base handler - defines comparison logic, training, and leaderboard"""
    
    def __init__(self, name: str):
        self.name = name
        self.battles: List[Battle] = []
        self.elos: Dict[str, float] = {}
    
    def get_label(self, acc_a: float, tokens_a: int, turns_a: int,
                  acc_b: float, tokens_b: int, turns_b: int) -> str:
        """Return label: 'a_wins', 'b_wins', 'tie', or 'both'"""
        raise NotImplementedError
    
    def loss_function(self, elo_a: float, elo_b: float, label: str) -> float:
        """Compute loss for this battle"""
        raise NotImplementedError
    
    def train(self, learning_rate: float = 0.01, num_iterations: int = 100) -> None:
        """Train model parameters (e.g., elos) from battles"""
        raise NotImplementedError
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Return sorted leaderboard with metrics"""
        raise NotImplementedError


class BTHandler(Handler):
    """Classic Bradley-Terry: accuracy > tokens > turns"""
    
    def get_label(self, acc_a: float, tokens_a: int, turns_a: int,
                  acc_b: float, tokens_b: int, turns_b: int) -> str:
        # Compare accuracy
        if acc_a > acc_b:
            return "a_wins"
        elif acc_b > acc_a:
            return "b_wins"
        
        # Tiebreak by tokens (lower is better)
        if tokens_a < tokens_b:
            return "a_wins"
        elif tokens_b < tokens_a:
            return "b_wins"
        
        # Tiebreak by turns (lower is better)
        if turns_a < turns_b:
            return "a_wins"
        elif turns_b < turns_a:
            return "b_wins"
        
        # Still tied - return both
        return "both"
    
    def loss_function(self, elo_a: float, elo_b: float, label: str) -> float:
        """Bradley-Terry loss"""
        # Predicted probability a wins
        p_a_wins = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))
        
        # True outcome
        if label == "a_wins":
            y_true = 1.0
        elif label == "b_wins":
            y_true = 0.0
        elif label == "tie":
            y_true = 0.5
        else:  # both
            y_true = 0.5
        
        # Cross-entropy loss
        eps = 1e-10
        return -(y_true * np.log(p_a_wins + eps) + (1 - y_true) * np.log(1 - p_a_wins + eps))
    
    def train(self, learning_rate: float = 32.0, num_iterations: int = 100) -> None:
        """Train elos using gradient descent"""
        # Initialize elos to 1500
        models = set()
        for battle in self.battles:
            models.add(battle.model_a)
            models.add(battle.model_b)
        
        self.elos = {model: 1500.0 for model in models}
        
        # Gradient descent
        for iteration in range(num_iterations):
            gradients = {model: 0.0 for model in models}
            
            for battle in self.battles:
                elo_a = self.elos[battle.model_a]
                elo_b = self.elos[battle.model_b]
                
                # Predicted
                p_a_wins = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))
                
                # True
                if battle.label == "a_wins":
                    y_true = 1.0
                elif battle.label == "b_wins":
                    y_true = 0.0
                else:  # tie or both
                    y_true = 0.5
                
                # Gradient
                error = p_a_wins - y_true
                grad = error * (np.log(10) / 400) * p_a_wins * (1 - p_a_wins)
                
                gradients[battle.model_a] += grad
                gradients[battle.model_b] -= grad
            
            # Update
            for model in models:
                self.elos[model] -= learning_rate * gradients[model]
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Return models sorted by elo"""
        leaderboard = [
            {"model": model, "elo": elo}
            for model, elo in self.elos.items()
        ]
        leaderboard.sort(key=lambda x: x["elo"], reverse=True)
        
        # Add ranks
        for i, entry in enumerate(leaderboard, 1):
            entry["rank"] = i
        
        return leaderboard


class Evaluator:
    """Main evaluator - manages handlers and battle generation"""
    
    def __init__(self, data: Dict[str, Dict[str, List[Attempt]]]):
        """
        Args:
            data: {model_id: {question_id: [Attempt, ...]}}
        """
        self.data = data
        self.handlers: Dict[str, Handler] = {}
    
    def register_handler(self, handler: Handler) -> None:
        """Register a new handler"""
        self.handlers[handler.name] = handler
    
    def generate_battles(self, num_battles: int, seed: int = None) -> None:
        """Generate battles by sampling models/questions/attempts"""
        if seed is not None:
            random.seed(seed)
        
        models = list(self.data.keys())
        
        for _ in range(num_battles):
            # Sample two different models
            if len(models) < 2:
                continue
            
            model_a, model_b = random.sample(models, 2)
            
            # Find common questions
            questions_a = set(self.data[model_a].keys())
            questions_b = set(self.data[model_b].keys())
            common_questions = list(questions_a & questions_b)
            
            if not common_questions:
                continue
            
            # Sample question
            question_id = random.choice(common_questions)
            
            # Sample random attempt from each
            attempt_a = random.choice(self.data[model_a][question_id])
            attempt_b = random.choice(self.data[model_b][question_id])
            
            # Get label from each handler
            for handler in self.handlers.values():
                label = handler.get_label(
                    attempt_a.accuracy, attempt_a.num_tokens, attempt_a.num_turns,
                    attempt_b.accuracy, attempt_b.num_tokens, attempt_b.num_turns
                )
                
                battle = Battle(
                    model_a=model_a,
                    model_b=model_b,
                    question_id=question_id,
                    attempt_a=attempt_a,
                    attempt_b=attempt_b,
                    label=label
                )
                
                handler.battles.append(battle)
    
    def train_all(self, **train_kwargs) -> None:
        """Train all handlers"""
        for handler in self.handlers.values():
            handler.train(**train_kwargs)
    
    def get_leaderboard(self, handler_name: str) -> List[Dict[str, Any]]:
        """Get leaderboard for a specific handler"""
        if handler_name not in self.handlers:
            raise ValueError(f"Handler {handler_name} not found")
        
        return self.handlers[handler_name].get_leaderboard()
    
    def print_leaderboard(self, handler_name: str) -> None:
        """Print leaderboard"""
        leaderboard = self.get_leaderboard(handler_name)
        
        print(f"\n{'='*60}")
        print(f"LEADERBOARD: {handler_name}")
        print(f"{'='*60}")
        print(f"{'Rank':<6} {'Model':<30} {'Elo':<10}")
        print(f"{'-'*60}")
        
        for entry in leaderboard:
            print(f"{entry['rank']:<6} {entry['model']:<30} {entry['elo']:<10.1f}")
        
        print(f"{'='*60}\n")

