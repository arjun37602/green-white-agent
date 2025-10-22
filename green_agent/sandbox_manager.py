"""
Sandbox Environment Manager for TerminalBench

This module provides sandbox management capabilities for the Green Agent,
enabling isolated execution environments for terminal tasks.
"""

import os
import subprocess
import tempfile
import shutil
import logging
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CommandResult:
    """Result of a command execution in sandbox"""
    command: str
    stdout: str
    stderr: str
    returncode: int
    execution_time: float
    timestamp: str
    success: bool


@dataclass
class SandboxState:
    """Captured state of a sandbox environment"""
    sandbox_id: str
    working_directory: str
    environment_vars: Dict[str, str]
    file_system_snapshot: Dict[str, Any]
    timestamp: str


class SandboxManager:
    """Manages isolated terminal environments for task execution"""
    
    def __init__(self, sandbox_type: str = "directory", base_path: Optional[str] = None):
        self.sandbox_type = sandbox_type
        self.base_path = base_path or tempfile.mkdtemp(prefix="terminalbench_")
        self.active_sandboxes: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Ensure base path exists
        os.makedirs(self.base_path, exist_ok=True)
        self.logger.info(f"SandboxManager initialized with base path: {self.base_path}")
    
    def create_sandbox(self, task_id: str, environment_spec: Optional[Dict[str, Any]] = None) -> str:
        """
        Create an isolated environment for task execution.
        
        Args:
            task_id: Unique identifier for the task
            environment_spec: Environment configuration (working directory, etc.)
            
        Returns:
            Sandbox ID for referencing this environment
        """
        sandbox_id = f"sandbox_{task_id}_{int(time.time())}"
        sandbox_path = os.path.join(self.base_path, sandbox_id)
        
        try:
            # Create sandbox directory
            os.makedirs(sandbox_path, exist_ok=True)
            
            # Set up working directory
            working_dir = environment_spec.get("working_directory", "/app") if environment_spec else "/app"
            if working_dir.startswith("/"):
                # Convert absolute path to relative within sandbox
                working_dir = working_dir.lstrip("/")
            
            sandbox_working_dir = os.path.join(sandbox_path, working_dir)
            os.makedirs(sandbox_working_dir, exist_ok=True)
            
            # Store sandbox information
            self.active_sandboxes[sandbox_id] = {
                "path": sandbox_path,
                "working_directory": sandbox_working_dir,
                "environment_spec": environment_spec or {},
                "created_at": datetime.utcnow().isoformat(),
                "command_history": [],
                "file_snapshots": []
            }
            
            self.logger.info(f"Created sandbox {sandbox_id} at {sandbox_path}")
            return sandbox_id
            
        except Exception as e:
            self.logger.error(f"Failed to create sandbox {sandbox_id}: {e}")
            raise
    
    def execute_command(self, sandbox_id: str, command: str, timeout: int = 30) -> CommandResult:
        """
        Execute a command within the specified sandbox.
        
        Args:
            sandbox_id: ID of the sandbox to execute in
            command: Command to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            CommandResult with execution details
        """
        if sandbox_id not in self.active_sandboxes:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        
        sandbox_info = self.active_sandboxes[sandbox_id]
        working_dir = sandbox_info["working_directory"]
        
        start_time = time.time()
        timestamp = datetime.utcnow().isoformat()
        
        try:
            self.logger.info(f"Executing command in sandbox {sandbox_id}: {command}")
            
            # Execute command in sandbox working directory
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=working_dir,
                timeout=timeout,
                env=self._get_sandbox_env(sandbox_id)
            )
            
            execution_time = time.time() - start_time
            
            command_result = CommandResult(
                command=command,
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
                execution_time=execution_time,
                timestamp=timestamp,
                success=result.returncode == 0
            )
            
            # Store command in history
            sandbox_info["command_history"].append(command_result)
            
            self.logger.info(f"Command completed in {execution_time:.2f}s with return code {result.returncode}")
            return command_result
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            self.logger.warning(f"Command timed out after {timeout}s: {command}")
            
            return CommandResult(
                command=command,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                returncode=-1,
                execution_time=execution_time,
                timestamp=timestamp,
                success=False
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Command execution failed: {e}")
            
            return CommandResult(
                command=command,
                stdout="",
                stderr=str(e),
                returncode=-1,
                execution_time=execution_time,
                timestamp=timestamp,
                success=False
            )
    
    def capture_state(self, sandbox_id: str) -> SandboxState:
        """
        Capture current state of the sandbox for reproducibility.
        
        Args:
            sandbox_id: ID of the sandbox to capture
            
        Returns:
            SandboxState with current environment state
        """
        if sandbox_id not in self.active_sandboxes:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        
        sandbox_info = self.active_sandboxes[sandbox_id]
        
        # Capture file system state
        file_snapshot = self._capture_file_system(sandbox_info["path"])
        
        # Store snapshot
        sandbox_info["file_snapshots"].append(file_snapshot)
        
        return SandboxState(
            sandbox_id=sandbox_id,
            working_directory=sandbox_info["working_directory"],
            environment_vars=self._get_sandbox_env(sandbox_id),
            file_system_snapshot=file_snapshot,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def reset_sandbox(self, sandbox_id: str, to_state: Optional[SandboxState] = None):
        """
        Reset sandbox to clean state or to a specific captured state.
        
        Args:
            sandbox_id: ID of the sandbox to reset
            to_state: Optional state to reset to (if None, resets to clean state)
        """
        if sandbox_id not in self.active_sandboxes:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        
        sandbox_info = self.active_sandboxes[sandbox_id]
        
        if to_state:
            # Reset to specific state
            self._restore_file_system(sandbox_info["path"], to_state.file_system_snapshot)
            self.logger.info(f"Reset sandbox {sandbox_id} to captured state")
        else:
            # Reset to clean state
            self._clean_sandbox(sandbox_id)
            self.logger.info(f"Reset sandbox {sandbox_id} to clean state")
    
    def destroy_sandbox(self, sandbox_id: str):
        """
        Clean up sandbox resources.
        
        Args:
            sandbox_id: ID of the sandbox to destroy
        """
        if sandbox_id not in self.active_sandboxes:
            self.logger.warning(f"Sandbox {sandbox_id} not found for destruction")
            return
        
        sandbox_info = self.active_sandboxes[sandbox_id]
        
        try:
            # Remove sandbox directory
            shutil.rmtree(sandbox_info["path"], ignore_errors=True)
            
            # Remove from active sandboxes
            del self.active_sandboxes[sandbox_id]
            
            self.logger.info(f"Destroyed sandbox {sandbox_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to destroy sandbox {sandbox_id}: {e}")
    
    def get_sandbox_info(self, sandbox_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a sandbox."""
        return self.active_sandboxes.get(sandbox_id)
    
    def list_sandboxes(self) -> List[str]:
        """List all active sandbox IDs."""
        return list(self.active_sandboxes.keys())
    
    def cleanup_all(self):
        """Clean up all sandboxes and base directory."""
        for sandbox_id in list(self.active_sandboxes.keys()):
            self.destroy_sandbox(sandbox_id)
        
        # Remove base directory
        try:
            shutil.rmtree(self.base_path, ignore_errors=True)
            self.logger.info("Cleaned up all sandbox resources")
        except Exception as e:
            self.logger.error(f"Failed to clean up base directory: {e}")
    
    def _get_sandbox_env(self, sandbox_id: str) -> Dict[str, str]:
        """Get environment variables for sandbox execution."""
        env = os.environ.copy()
        env["SANDBOX_ID"] = sandbox_id
        env["SANDBOX_PATH"] = self.active_sandboxes[sandbox_id]["path"]
        return env
    
    def _capture_file_system(self, sandbox_path: str) -> Dict[str, Any]:
        """Capture file system state of sandbox."""
        snapshot = {
            "files": {},
            "directories": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            for root, dirs, files in os.walk(sandbox_path):
                # Record directories
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    rel_path = os.path.relpath(dir_path, sandbox_path)
                    snapshot["directories"].append(rel_path)
                
                # Record files
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    rel_path = os.path.relpath(file_path, sandbox_path)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        snapshot["files"][rel_path] = content
                    except Exception:
                        # Skip binary files or files that can't be read
                        snapshot["files"][rel_path] = "<binary_file>"
        
        except Exception as e:
            self.logger.error(f"Failed to capture file system: {e}")
        
        return snapshot
    
    def _restore_file_system(self, sandbox_path: str, snapshot: Dict[str, Any]):
        """Restore file system state from snapshot."""
        try:
            # Clean existing files
            self._clean_sandbox_directory(sandbox_path)
            
            # Restore directories
            for dir_path in snapshot["directories"]:
                full_path = os.path.join(sandbox_path, dir_path)
                os.makedirs(full_path, exist_ok=True)
            
            # Restore files
            for file_path, content in snapshot["files"].items():
                if content != "<binary_file>":
                    full_path = os.path.join(sandbox_path, file_path)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
        
        except Exception as e:
            self.logger.error(f"Failed to restore file system: {e}")
    
    def _clean_sandbox(self, sandbox_id: str):
        """Clean sandbox to initial state."""
        sandbox_info = self.active_sandboxes[sandbox_id]
        self._clean_sandbox_directory(sandbox_info["path"])
        
        # Recreate working directory
        os.makedirs(sandbox_info["working_directory"], exist_ok=True)
        
        # Clear command history
        sandbox_info["command_history"] = []
        sandbox_info["file_snapshots"] = []
    
    def _clean_sandbox_directory(self, sandbox_path: str):
        """Clean all contents of sandbox directory."""
        try:
            for root, dirs, files in os.walk(sandbox_path, topdown=False):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
                
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        os.rmdir(dir_path)
                    except Exception:
                        pass
        except Exception as e:
            self.logger.error(f"Failed to clean sandbox directory: {e}")
