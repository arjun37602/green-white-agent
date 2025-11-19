"""
Minimal evaluation system with handler registry.
Handlers define how to compare models (get_label), train (loss), and report (leaderboard).
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any
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
    
    def train(self, **kwargs) -> None:
        """Train model parameters (e.g., elos) from battles"""
        raise NotImplementedError
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Return sorted leaderboard with metrics"""
        raise NotImplementedError


class BradleyTerryModel(nn.Module):
    """PyTorch model for Bradley-Terry ratings"""
    
    def __init__(self, num_models: int, init_mean: float = 1500.0, init_std: float = 50.0):
        super().__init__()
        self.elos = nn.Parameter(torch.randn(num_models) * init_std + init_mean)
    
    def forward(self, idx_a: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
        """Compute P(A wins over B) using Bradley-Terry model"""
        elo_a = self.elos[idx_a]
        elo_b = self.elos[idx_b]
        elo_diff = (elo_b - elo_a) / 400.0
        p_a_wins = torch.sigmoid(-elo_diff * torch.log(torch.tensor(10.0)))
        return p_a_wins


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
        
        return "both"
    
    def train(self, learning_rate: float = 0.1, num_epochs: int = 1000, 
              init_mean: float = 1500.0, init_std: float = 50.0) -> None:
        """Train elos using PyTorch"""
        # Get unique models
        models = set()
        for battle in self.battles:
            models.add(battle.model_a)
            models.add(battle.model_b)
        
        model_list = sorted(list(models))
        model_to_idx = {model: i for i, model in enumerate(model_list)}
        
        # Create BT model
        torch.manual_seed(42)
        bt_model = BradleyTerryModel(len(model_list), init_mean, init_std)
        optimizer = optim.Adam(bt_model.parameters(), lr=learning_rate)
        
        # Prepare battle tensors
        indices_a = []
        indices_b = []
        labels = []
        
        for battle in self.battles:
            indices_a.append(model_to_idx[battle.model_a])
            indices_b.append(model_to_idx[battle.model_b])
            
            if battle.label == "a_wins":
                labels.append(1.0)
            elif battle.label == "b_wins":
                labels.append(0.0)
            else:
                labels.append(0.5)
        
        idx_a_tensor = torch.tensor(indices_a)
        idx_b_tensor = torch.tensor(indices_b)
        y_true_tensor = torch.tensor(labels)
        
        # Training loop
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            p_a_wins = bt_model(idx_a_tensor, idx_b_tensor)
            
            # Binary cross-entropy loss
            loss = -(y_true_tensor * torch.log(p_a_wins + 1e-10) + 
                    (1 - y_true_tensor) * torch.log(1 - p_a_wins + 1e-10)).mean()
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Save final elos
        with torch.no_grad():
            self.elos = {model: bt_model.elos[model_to_idx[model]].item() 
                        for model in model_list}
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Return models sorted by elo"""
        leaderboard = [
            {"model": model, "elo": elo}
            for model, elo in self.elos.items()
        ]
        leaderboard.sort(key=lambda x: x["elo"], reverse=True)
        
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
