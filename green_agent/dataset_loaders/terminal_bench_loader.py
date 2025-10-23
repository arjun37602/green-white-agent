#!/usr/bin/env python3
"""
Terminal-Bench Task Loader

This module loads and parses Terminal-Bench tasks from the official dataset format.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TerminalBenchTask:
    """Represents a Terminal-Bench task with all its components."""
    task_id: str
    task_path: str
    instruction: str
    author_name: str
    author_email: str
    difficulty: str
    category: str
    tags: List[str]
    parser_name: str
    max_agent_timeout_sec: float
    max_test_timeout_sec: float
    run_tests_in_same_shell: bool
    expert_time_estimate_min: Optional[float]
    junior_time_estimate_min: Optional[float]
    solution_commands: List[str]
    test_file_path: str
    run_tests_script_path: str
    dockerfile_path: str
    docker_compose_path: str
    task_deps_dir: Optional[str]


class TerminalBenchTaskLoader:
    """Loads Terminal-Bench tasks from dataset directories."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize the loader with a Terminal-Bench dataset path.
        
        Args:
            dataset_path: Path to the Terminal-Bench dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.logger = logging.getLogger(__name__)
        
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        self.logger.info(f"TerminalBenchTaskLoader initialized with dataset path: {self.dataset_path}")
    
    def load_task(self, task_id: str) -> TerminalBenchTask:
        """
        Load a single Terminal-Bench task by ID.
        
        Args:
            task_id: The task ID (directory name)
            
        Returns:
            TerminalBenchTask object
        """
        task_path = self.dataset_path / task_id
        
        if not task_path.exists():
            raise ValueError(f"Task not found: {task_id}")
        
        self.logger.info(f"Loading Terminal-Bench task: {task_id}")
        
        # Parse task.yaml
        task_yaml_path = task_path / "task.yaml"
        if not task_yaml_path.exists():
            raise ValueError(f"task.yaml not found for task: {task_id}")
        
        with open(task_yaml_path, 'r') as f:
            task_data = yaml.safe_load(f)
        
        # Load solution commands
        solution_commands = self._load_solution_commands(task_path)
        
        # Get test file path
        test_file_path = task_path / "tests" / "test_outputs.py"
        if not test_file_path.exists():
            raise ValueError(f"test_outputs.py not found for task: {task_id}")
        
        # Get run-tests.sh path
        run_tests_script_path = task_path / "run-tests.sh"
        if not run_tests_script_path.exists():
            raise ValueError(f"run-tests.sh not found for task: {task_id}")
        
        # Get Dockerfile path
        dockerfile_path = task_path / "Dockerfile"
        if not dockerfile_path.exists():
            raise ValueError(f"Dockerfile not found for task: {task_id}")
        
        # Get docker-compose.yaml path
        docker_compose_path = task_path / "docker-compose.yaml"
        if not docker_compose_path.exists():
            raise ValueError(f"docker-compose.yaml not found for task: {task_id}")
        
        # Check for task dependencies
        task_deps_dir = task_path / "task-deps"
        task_deps_path = str(task_deps_dir) if task_deps_dir.exists() else None
        
        # Create TerminalBenchTask object
        task = TerminalBenchTask(
            task_id=task_id,
            task_path=str(task_path),
            instruction=task_data.get('instruction', ''),
            author_name=task_data.get('author_name', ''),
            author_email=task_data.get('author_email', ''),
            difficulty=task_data.get('difficulty', 'unknown'),
            category=task_data.get('category', ''),
            tags=task_data.get('tags', []),
            parser_name=task_data.get('parser_name', 'pytest'),
            max_agent_timeout_sec=task_data.get('max_agent_timeout_sec', 900.0),
            max_test_timeout_sec=task_data.get('max_test_timeout_sec', 180.0),
            run_tests_in_same_shell=task_data.get('run_tests_in_same_shell', False),
            expert_time_estimate_min=task_data.get('expert_time_estimate_min'),
            junior_time_estimate_min=task_data.get('junior_time_estimate_min'),
            solution_commands=solution_commands,
            test_file_path=str(test_file_path),
            run_tests_script_path=str(run_tests_script_path),
            dockerfile_path=str(dockerfile_path),
            docker_compose_path=str(docker_compose_path),
            task_deps_dir=task_deps_path
        )
        
        self.logger.info(f"Successfully loaded task {task_id}: {task.instruction[:100]}...")
        return task
    
    def load_tasks_from_dataset(self, task_ids: Optional[List[str]] = None, limit: Optional[int] = None) -> List[TerminalBenchTask]:
        """
        Load multiple Terminal-Bench tasks from the dataset.
        
        Args:
            task_ids: Optional list of specific task IDs to load
            limit: Optional limit on number of tasks to load
            
        Returns:
            List of TerminalBenchTask objects
        """
        if task_ids:
            # Load specific tasks
            tasks = []
            for task_id in task_ids:
                try:
                    task = self.load_task(task_id)
                    tasks.append(task)
                except Exception as e:
                    self.logger.error(f"Failed to load task {task_id}: {e}")
            return tasks
        else:
            # Load all tasks in dataset
            all_task_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]
            
            tasks = []
            for task_dir in all_task_dirs:
                try:
                    task = self.load_task(task_dir.name)
                    tasks.append(task)
                    
                    if limit and len(tasks) >= limit:
                        break
                        
                except Exception as e:
                    self.logger.error(f"Failed to load task {task_dir.name}: {e}")
            
            self.logger.info(f"Loaded {len(tasks)} Terminal-Bench tasks from dataset")
            return tasks
    
    def _load_solution_commands(self, task_path: Path) -> List[str]:
        """Load solution commands from solution.sh or solution.yaml."""
        solution_sh_path = task_path / "solution.sh"
        solution_yaml_path = task_path / "solution.yaml"
        
        if solution_sh_path.exists():
            # Load from solution.sh
            with open(solution_sh_path, 'r') as f:
                content = f.read()
            
            # Parse bash commands (simple approach - split by lines and filter)
            commands = []
            for line in content.split('\n'):
                line = line.strip()
                # Skip empty lines, comments, and shebang
                if line and not line.startswith('#') and not line.startswith('#!/'):
                    commands.append(line)
            
            return commands
            
        elif solution_yaml_path.exists():
            # Load from solution.yaml
            with open(solution_yaml_path, 'r') as f:
                solution_data = yaml.safe_load(f)
            
            commands = []
            for item in solution_data:
                if isinstance(item, dict) and 'command' in item:
                    commands.append(item['command'])
                elif isinstance(item, str):
                    commands.append(item)
            
            return commands
        else:
            raise ValueError(f"No solution file found for task: {task_path}")
    
    def convert_to_internal_format(self, task: TerminalBenchTask) -> Dict[str, Any]:
        """
        Convert a TerminalBenchTask to our internal task format.
        
        Args:
            task: TerminalBenchTask object
            
        Returns:
            Dictionary in our internal task format
        """
        return {
            "id": task.task_id,
            "instruction": task.instruction,
            "test": f"Run pytest tests in {task.test_file_path}",
            "environment": {
                "working_directory": "/app",
                "max_agent_timeout_sec": task.max_agent_timeout_sec,
                "max_test_timeout_sec": task.max_test_timeout_sec,
                "run_tests_in_same_shell": task.run_tests_in_same_shell
            },
            "metadata": {
                "author_name": task.author_name,
                "author_email": task.author_email,
                "difficulty": task.difficulty,
                "category": task.category,
                "tags": task.tags,
                "parser_name": task.parser_name,
                "expert_time_estimate_min": task.expert_time_estimate_min,
                "junior_time_estimate_min": task.junior_time_estimate_min,
                "solution_commands": task.solution_commands,
                "test_file_path": task.test_file_path,
                "run_tests_script_path": task.run_tests_script_path,
                "dockerfile_path": task.dockerfile_path,
                "docker_compose_path": task.docker_compose_path,
                "task_deps_dir": task.task_deps_dir
            }
        }
    
    def list_available_tasks(self) -> List[str]:
        """
        List all available task IDs in the dataset.
        
        Returns:
            List of task IDs
        """
        task_dirs = [d.name for d in self.dataset_path.iterdir() if d.is_dir()]
        return sorted(task_dirs)