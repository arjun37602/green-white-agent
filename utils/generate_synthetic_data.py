#!/usr/bin/env python3
"""
Generate synthetic data matching green-white agent output format
"""

import sys
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from green_agent.attempt_store import AttemptStore, Attempt
from green_agent.evaluation import Evaluator, BTHandler


def generate_synthetic_data(
    models: list,
    questions: list,
    num_attempts: int,
    model_profiles: dict,
    output_dir: str = "./synthetic_results"
):
    """
    Generate synthetic data matching green-white agent output.
    
    Args:
        models: List of model IDs
        questions: List of question IDs
        num_attempts: Number of attempts per model per question
        model_profiles: Dict mapping model_id to performance profile
        output_dir: Where to store results
    """
    # Clean and create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    store = AttemptStore(output_dir)
    
    print(f"Generating synthetic data:")
    print(f"  Models: {len(models)}")
    print(f"  Questions: {len(questions)}")
    print(f"  Attempts per question: {num_attempts}")
    print(f"  Total attempts: {len(models) * len(questions) * num_attempts}")
    
    np.random.seed(42)
    
    for model_id in models:
        profile = model_profiles[model_id]
        
        for question_id in questions:
            for attempt_id in range(num_attempts):
                # Add noise to base stats
                acc = np.clip(
                    np.random.normal(profile["accuracy"], profile["acc_std"]),
                    0.0, 1.0
                )
                tokens = int(np.clip(
                    np.random.normal(profile["tokens"], profile["token_std"]),
                    100, 50000
                ))
                turns = int(np.clip(
                    np.random.normal(profile["turns"], profile["turn_std"]),
                    1, 20
                ))
                
                attempt = Attempt(
                    attempt_id=attempt_id,
                    accuracy=float(acc),
                    num_tokens=tokens,
                    num_turns=turns,
                    timestamp=datetime.utcnow().isoformat(),
                    metadata={
                        "synthetic": True,
                        "model_profile": profile
                    }
                )
                
                store.save_attempt(model_id, question_id, attempt)
    
    print(f"Data saved to {output_dir}/")
    return store


def print_data_summary(data: dict):
    """Print summary of loaded data"""
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    for model_id in sorted(data.keys()):
        questions_data = data[model_id]
        total_attempts = sum(len(attempts) for attempts in questions_data.values())
        
        print(f"\n{model_id}:")
        print(f"  Questions: {len(questions_data)}")
        print(f"  Total attempts: {total_attempts}")
        
        # Calculate average stats
        all_attempts = [att for attempts in questions_data.values() for att in attempts]
        if all_attempts:
            avg_acc = np.mean([a.accuracy for a in all_attempts])
            avg_tokens = np.mean([a.num_tokens for a in all_attempts])
            avg_turns = np.mean([a.num_turns for a in all_attempts])
            
            print(f"  Avg accuracy: {avg_acc:.3f}")
            print(f"  Avg tokens: {avg_tokens:.1f}")
            print(f"  Avg turns: {avg_turns:.1f}")


def save_leaderboard(leaderboard: list, output_path: str):
    """Save leaderboard to JSON"""
    with open(output_path, 'w') as f:
        json.dump(leaderboard, f, indent=2)
    print(f"Leaderboard saved to {output_path}")


def main():
    # Configuration
    models = ["gpt-4", "claude-3.5", "gpt-3.5-turbo", "gemini-pro", "mixtral-8x7b"]
    questions = [f"question_{i:03d}" for i in range(10)]
    num_attempts = 5
    
    # Model profiles (base performance + noise)
    model_profiles = {
        "gpt-4": {
            "accuracy": 0.90,
            "tokens": 6000,
            "turns": 3,
            "acc_std": 0.08,
            "token_std": 1200,
            "turn_std": 0.8
        },
        "claude-3.5": {
            "accuracy": 0.88,
            "tokens": 7000,
            "turns": 4,
            "acc_std": 0.10,
            "token_std": 1400,
            "turn_std": 1.0
        },
        "gpt-3.5-turbo": {
            "accuracy": 0.70,
            "tokens": 9000,
            "turns": 5,
            "acc_std": 0.15,
            "token_std": 1800,
            "turn_std": 1.2
        },
        "gemini-pro": {
            "accuracy": 0.85,
            "tokens": 6500,
            "turns": 3,
            "acc_std": 0.12,
            "token_std": 1300,
            "turn_std": 0.9
        },
        "mixtral-8x7b": {
            "accuracy": 0.75,
            "tokens": 8000,
            "turns": 4,
            "acc_std": 0.13,
            "token_std": 1600,
            "turn_std": 1.1
        }
    }
    
    output_dir = "./synthetic_results"
    
    print("="*60)
    print("SYNTHETIC DATA GENERATION & BT EVALUATION")
    print("="*60)
    
    # Generate data
    store = generate_synthetic_data(
        models, questions, num_attempts, model_profiles, output_dir
    )
    
    # Load data
    print("\nLoading data...")
    data = store.load_all()
    
    # Print summary
    print_data_summary(data)
    
    # Create evaluator
    print("\n" + "="*60)
    print("BRADLEY-TERRY EVALUATION")
    print("="*60)
    
    evaluator = Evaluator(data)
    bt_handler = BTHandler("classic_bt")
    evaluator.register_handler(bt_handler)
    print("Registered BT handler")
    
    # Generate battles
    num_battles = 500
    print(f"\nGenerating {num_battles} battles...")
    evaluator.generate_battles(num_battles, seed=42)
    print(f"Generated {len(bt_handler.battles)} battles")
    
    # Show sample battles
    print("\nSample battles:")
    for i, battle in enumerate(bt_handler.battles[:5], 1):
        print(f"  {i}. {battle.model_a} vs {battle.model_b} on {battle.question_id}")
        print(f"     {battle.model_a}: acc={battle.attempt_a.accuracy:.3f}, "
              f"tokens={battle.attempt_a.num_tokens}, turns={battle.attempt_a.num_turns}")
        print(f"     {battle.model_b}: acc={battle.attempt_b.accuracy:.3f}, "
              f"tokens={battle.attempt_b.num_tokens}, turns={battle.attempt_b.num_turns}")
        print(f"     → {battle.label}")
    
    # Train
    print("\nTraining BT model...")
    evaluator.train_all(
        learning_rate=0.1,
        num_epochs=1000,
        init_mean=1500.0,
        init_std=50.0
    )
    print("Training complete")
    
    # Get and print leaderboard
    evaluator.print_leaderboard("classic_bt")
    
    # Save leaderboard
    leaderboard = evaluator.get_leaderboard("classic_bt")
    leaderboard_path = os.path.join(output_dir, "leaderboard.json")
    save_leaderboard(leaderboard, leaderboard_path)
    
    # Bootstrap experiment
    print("\n" + "="*60)
    print("BOOTSTRAP EXPERIMENT")
    print("="*60)
    print("Running bootstrap with 50 attempts...")
    
    bootstrap_results = evaluator.bootstrap(
        "classic_bt", 
        num_battles=num_battles, 
        num_attempts=50,
        confidence_level=0.95,
        seed=42
    )
    
    print("\nBootstrap Results (95% CI):")
    sorted_results = sorted(bootstrap_results.items(), key=lambda x: x[1]['mean'], reverse=True)
    for model, stats in sorted_results:
        print(f"  {model:20s} Elo: {stats['mean']:6.1f} ± {stats['std']:5.1f}  "
              f"CI: [{stats['ci_lower']:6.1f}, {stats['ci_upper']:6.1f}]")
    
    # Save bootstrap results
    bootstrap_path = os.path.join(output_dir, "bootstrap_results.json")
    with open(bootstrap_path, 'w') as f:
        json.dump(bootstrap_results, f, indent=2)
    print(f"\nBootstrap results saved to {bootstrap_path}")
    
    # Save config
    config = {
        "models": models,
        "questions": questions,
        "num_attempts": num_attempts,
        "num_battles": num_battles,
        "bootstrap_attempts": 50,
        "model_profiles": model_profiles,
        "generated_at": datetime.utcnow().isoformat()
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}/")
    print(f"  - {output_dir}/[model]/[question]/attempts.json")
    print(f"  - {output_dir}/leaderboard.json")
    print(f"  - {output_dir}/bootstrap_results.json")
    print(f"  - {output_dir}/config.json")


if __name__ == "__main__":
    main()

