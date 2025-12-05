#!/usr/bin/env python3
"""
Terminal-Bench Task Loader

This module loads and parses Terminal-Bench tasks using the official Terminal Bench Dataset class.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from terminal_bench.dataset.dataset import Dataset
from terminal_bench.handlers.trial_handler import Task as TBTask, TaskPaths

logger = logging.getLogger(__name__)


class TerminalBenchTaskLoader:
    """Loads Terminal-Bench tasks using the official Dataset class."""
    
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
        
        # Initialize Terminal Bench Dataset with local path
        self.dataset = Dataset(path=self.dataset_path)
        
        self.logger.info(f"TerminalBenchTaskLoader initialized with dataset path: {self.dataset_path}")
    
    def load_task(self, task_id: str) -> tuple[TBTask, TaskPaths]:
        """
        Load a single Terminal-Bench task by ID using the official Dataset class.
        
        Args:
            task_id: The task ID (directory name)
            
        Returns:
            Tuple of (Task object, TaskPaths object)
            
        Raises:
            ValueError: If task not found
        """
        self.logger.info(f"Loading Terminal-Bench task: {task_id}")
        
        try:
            task_path = self.dataset.config.path / task_id
            task_paths = TaskPaths(task_path)  # TaskPaths constructor takes input_path directly
            task = TBTask.from_yaml(task_paths.task_config_path)  # Access property as attribute, not method
            
            self.logger.info(f"Successfully loaded task {task_id}")
            return task, task_paths
            
        except Exception as e:
            raise ValueError(f"Failed to load task {task_id}: {e}")
    
    def load_tasks_from_dataset(self, task_ids: Optional[List[str]] = None) -> List[tuple[TBTask, TaskPaths]]:
        """
        Load multiple Terminal-Bench tasks from the dataset.
        
        Args:
            task_ids: Optional list of specific task IDs to load
            limit: Optional limit on number of tasks to load
            
        Returns:
            List of (Task, TaskPaths) tuples
        """
        tasks = []
        
        if task_ids:
            # Load specific tasks
            for task_id in task_ids:
                try:
                    task, task_paths = self.load_task(task_id)
                    tasks.append((task, task_paths))
                except Exception as e:
                    self.logger.error(f"Failed to load task {task_id}: {e}")
        else:
            # Load all tasks in dataset
            for task_path in self.dataset:
                try:
                    task_id = task_path.name
                    task, task_paths = self.load_task(task_id)
                    tasks.append((task, task_paths))
                    
                        
                except Exception as e:
                    self.logger.error(f"Failed to load task {task_path.name}: {e}")
        
        self.logger.info(f"Loaded {len(tasks)} Terminal-Bench tasks from dataset")
        return tasks