#!/usr/bin/env python3
"""
Terminal-Bench Task Loader

This module loads and parses Terminal-Bench tasks using the official Terminal Bench Dataset class.
"""

import json
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
        Auto-downloads dataset if not present.
        
        Args:
            dataset_path: Path to the Terminal-Bench dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.logger = logging.getLogger(__name__)
        
        if not self.dataset_path.exists():
            self.logger.warning(f"Dataset path does not exist: {dataset_path}")
            self.logger.info("Attempting to auto-download Terminal-Bench dataset...")
            
            try:
                # Try to auto-download by initializing Dataset with name and version
                # This will download to default cache location
                self.dataset = Dataset(name="terminal-bench-core", version="0.1.1")
                self.dataset_path = self.dataset.config.path
                self.logger.info(f"Auto-downloaded dataset to: {self.dataset_path}")
            except Exception as e:
                raise ValueError(
                    f"Dataset path does not exist: {dataset_path}\n"
                    f"Auto-download failed: {e}\n"
                    f"Please download manually with: python -c 'from terminal_bench.dataset.dataset import Dataset; Dataset(name=\"terminal-bench-core\", version=\"0.1.1\")'"
                )
        else:
            # Initialize Terminal Bench Dataset with local path
            self.dataset = Dataset(path=self.dataset_path)
        
        # Try to load cached task names for efficiency
        self._cached_task_names = self._load_cached_task_names()
        
        self.logger.info(f"TerminalBenchTaskLoader initialized with dataset path: {self.dataset_path}")
    
    def _load_cached_task_names(self) -> Optional[List[str]]:
        """Load cached task names if available."""
        cache_file = Path(__file__).parent.parent / "task_names_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)
                    task_names = cache_data.get("task_ids", [])
                    self.logger.info(f"Loaded {len(task_names)} cached task names")
                    return task_names
            except Exception as e:
                self.logger.warning(f"Failed to load cached task names: {e}")
        return None
    
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
                    self.logger.warning(f"Task {task_id} not found in dataset, skipping: {e}")
        else:
            # Load all tasks - use cached names if available for efficiency
            if self._cached_task_names:
                self.logger.info(f"Using cached task names ({len(self._cached_task_names)} tasks)")
                for task_id in self._cached_task_names:
                    try:
                        task, task_paths = self.load_task(task_id)
                        tasks.append((task, task_paths))
                    except Exception as e:
                        self.logger.debug(f"Task {task_id} not in dataset, skipping")
            else:
                # Fallback: iterate through dataset
                self.logger.info("No cached task names, iterating through dataset")
                for task_path in self.dataset:
                    try:
                        task_id = task_path.name
                        task, task_paths = self.load_task(task_id)
                        tasks.append((task, task_paths))
                    except Exception as e:
                        self.logger.debug(f"Task {task_path.name} not in dataset, skipping")
        
        if self._cached_task_names and len(tasks) < len(self._cached_task_names):
            skipped = len(self._cached_task_names) - len(tasks)
            self.logger.info(f"Loaded {len(tasks)} Terminal-Bench tasks from dataset ({skipped} tasks not found in current dataset)")
        else:
            self.logger.info(f"Loaded {len(tasks)} Terminal-Bench tasks from dataset")
        return tasks