#!/usr/bin/env python3
"""
Terminal Bench to A2A Converter

This script converts terminal bench problems to A2A-compatible format
and can send them to A2A agents for processing.
"""

import json
import requests
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
# Simple dataset loader (replacing terminal-bench dependency)
class SimpleDatasetLoader:
    """Simple dataset loader for JSON/JSONL files."""
    
    def load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Load dataset from JSON/JSONL file."""
        import json
        from pathlib import Path
        
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        elif path.suffix == '.jsonl':
            data = []
            with open(path, 'r') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return {
            "data": data if isinstance(data, list) else [data],
            "metadata": {
                "file_path": str(path),
                "num_records": len(data) if isinstance(data, list) else 1
            }
        }

class TerminalBenchToA2AConverter:
    """Convert terminal bench problems to A2A format and send to agents."""
    
    def __init__(self, a2a_agent_url: str = "http://localhost:8001"):
        self.a2a_agent_url = a2a_agent_url
        self.logger = logging.getLogger(__name__)
        self.loader = SimpleDatasetLoader()
    
    def convert_problem_to_a2a_task(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a terminal bench problem to A2A task format.
        
        Args:
            problem: Terminal bench problem dictionary
            
        Returns:
            A2A task dictionary
        """
        # Extract problem components
        instruction = problem.get("instruction", "")
        environment = problem.get("environment", {})
        test = problem.get("test", "")
        
        # Create problem description
        problem_description = f"""Terminal Bench Problem:

INSTRUCTION:
{instruction}

ENVIRONMENT:
{json.dumps(environment, indent=2) if environment else "No specific environment requirements"}

TEST:
{test}

Please provide a step-by-step solution for this terminal problem."""
        
        # Create A2A task
        a2a_task = {
            "artifacts": [
                {
                    "parts": [
                        {
                            "type": "text",
                            "text": problem_description
                        }
                    ]
                }
            ],
            "metadata": {
                "problem_id": problem.get("id", "unknown"),
                "difficulty": problem.get("difficulty", "unknown"),
                "category": problem.get("category", "unknown"),
                "tags": problem.get("tags", [])
            }
        }
        
        return a2a_task
    
    def send_task_to_a2a_agent(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send A2A task to an A2A agent.
        
        Args:
            task: A2A task dictionary
            
        Returns:
            Response from the agent
        """
        try:
            # Try both endpoints for compatibility
            endpoints = ["/tasks", "/tasks/"]
            
            for endpoint in endpoints:
                try:
                    response = requests.post(
                        f"{self.a2a_agent_url}{endpoint}",
                        json=task,
                        headers={"Content-Type": "application/json"},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                    elif response.status_code == 404 and endpoint == "/tasks":
                        # Try the alternative endpoint
                        continue
                    else:
                        self.logger.error(f"Agent returned status {response.status_code}: {response.text}")
                        return {"error": f"Agent error: {response.status_code}"}
                        
                except requests.exceptions.RequestException as e:
                    if endpoint == "/tasks":
                        # Try the alternative endpoint
                        continue
                    else:
                        raise e
            
            # If we get here, both endpoints failed
            return {"error": "All endpoints failed"}
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to send task to agent: {e}")
            return {"error": f"Request failed: {str(e)}"}
    
    def check_agent_health(self) -> bool:
        """Check if the A2A agent is running and healthy."""
        try:
            response = requests.get(
                f"{self.a2a_agent_url}/health",
                timeout=10
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_agent_card(self) -> Dict[str, Any]:
        """Get the agent card from the A2A agent."""
        try:
            response = requests.get(
                f"{self.a2a_agent_url}/agent-card",
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get agent card: {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def process_terminal_bench_dataset(self, dataset_path: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process a terminal bench dataset and convert problems to A2A tasks.
        
        Args:
            dataset_path: Path to terminal bench dataset
            output_path: Optional path to save A2A tasks
            
        Returns:
            List of A2A task results
        """
        # Load the dataset
        dataset = self.loader.load_dataset(dataset_path)
        problems = dataset.get("data", [])
        
        self.logger.info(f"Processing {len(problems)} problems from {dataset_path}")
        
        results = []
        
        for i, problem in enumerate(problems):
            self.logger.info(f"Processing problem {i+1}/{len(problems)}")
            
            # Convert to A2A task
            a2a_task = self.convert_problem_to_a2a_task(problem)
            
            # Send to agent (if agent is available)
            try:
                agent_response = self.send_task_to_a2a_agent(a2a_task)
                
                result = {
                    "problem_id": problem.get("id", f"problem_{i}"),
                    "original_problem": problem,
                    "a2a_task": a2a_task,
                    "agent_response": agent_response
                }
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process problem {i+1}: {e}")
                result = {
                    "problem_id": problem.get("id", f"problem_{i}"),
                    "original_problem": problem,
                    "a2a_task": a2a_task,
                    "agent_response": {"error": str(e)}
                }
                results.append(result)
        
        # Save results if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to {output_path}")
        
        return results
    
    def create_a2a_test_suite(self, dataset_path: str, output_path: str):
        """
        Create a test suite of A2A tasks from terminal bench dataset.
        
        Args:
            dataset_path: Path to terminal bench dataset
            output_path: Path to save A2A test suite
        """
        # Load the dataset
        dataset = self.loader.load_dataset(dataset_path)
        problems = dataset.get("data", [])
        
        # Convert all problems to A2A tasks
        a2a_tasks = []
        for problem in problems:
            a2a_task = self.convert_problem_to_a2a_task(problem)
            a2a_tasks.append({
                "id": problem.get("id", f"task_{len(a2a_tasks)}"),
                "task": a2a_task,
                "expected_fields": ["instruction", "environment", "test"],
                "metadata": {
                    "difficulty": problem.get("difficulty", "unknown"),
                    "category": problem.get("category", "unknown"),
                    "tags": problem.get("tags", [])
                }
            })
        
        # Save A2A test suite
        test_suite = {
            "name": "Terminal Bench A2A Test Suite",
            "description": "A2A-compatible test suite converted from Terminal Bench dataset",
            "version": "1.0.0",
            "tasks": a2a_tasks,
            "metadata": {
                "source_dataset": dataset_path,
                "total_tasks": len(a2a_tasks),
                "conversion_date": "2024-01-01"
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(test_suite, f, indent=2)
        
        self.logger.info(f"A2A test suite created with {len(a2a_tasks)} tasks: {output_path}")

def main():
    """Main function to demonstrate the converter."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Terminal Bench to A2A format")
    parser.add_argument("dataset_path", help="Path to terminal bench dataset")
    parser.add_argument("--output", "-o", help="Output path for A2A tasks")
    parser.add_argument("--agent-url", default="http://localhost:8001", help="A2A agent URL")
    parser.add_argument("--create-test-suite", action="store_true", help="Create A2A test suite")
    parser.add_argument("--check-agent", action="store_true", help="Check if agent is running")
    parser.add_argument("--agent-card", action="store_true", help="Get agent card")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create converter
    converter = TerminalBenchToA2AConverter(args.agent_url)
    
    if args.check_agent:
        print(f"🔍 Checking agent health at {args.agent_url}...")
        if converter.check_agent_health():
            print("✅ Agent is healthy and running")
        else:
            print("❌ Agent is not responding")
            print("💡 Start the agent with: python white_agent/agent.py --server")
        return
    
    if args.agent_card:
        print(f"📋 Getting agent card from {args.agent_url}...")
        card = converter.get_agent_card()
        if "error" in card:
            print(f"❌ Failed to get agent card: {card['error']}")
        else:
            print("✅ Agent card retrieved:")
            print(json.dumps(card, indent=2))
        return
    
    if args.create_test_suite:
        output_path = args.output or "terminal_bench_a2a_test_suite.json"
        converter.create_a2a_test_suite(args.dataset_path, output_path)
    else:
        # Check agent health before processing
        print(f"🔍 Checking agent health at {args.agent_url}...")
        if not converter.check_agent_health():
            print("⚠️  Agent is not responding. Processing will continue but tasks won't be sent to agent.")
            print("💡 Start the agent with: python white_agent/agent.py --server")
        
        # Process dataset and send to agent
        results = converter.process_terminal_bench_dataset(args.dataset_path, args.output)
        print(f"Processed {len(results)} problems")

if __name__ == "__main__":
    main()
