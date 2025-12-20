"""
Sandbox Environment Manager for TerminalBench

Docker-only sandbox management for isolated task execution.
Matches Terminal Bench's containerization approach.
"""

import os
import subprocess
import tempfile
import logging
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone

import docker
import docker.errors
from docker.models.containers import Container


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
    """Docker-based sandbox manager for isolated task execution"""
    
    CONTAINER_LOGS_PATH = "/logs"
    CONTAINER_AGENT_LOGS_PATH = "/agent-logs"
    CONTAINER_TEST_DIR = "/tests"
    CONTAINER_WORK_DIR = "/app"
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize Docker-based sandbox manager.
        
        Args:
            base_path: Base path for logs and metadata (default: temp directory)
        """
        self.base_path = base_path or tempfile.mkdtemp(prefix="terminalbench_")
        self.active_sandboxes: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
            self.logger.info("Docker client initialized successfully")
        except docker.errors.DockerException as e:
            raise RuntimeError(
                f"Docker not available: {e}\n"
                "Please ensure Docker is installed and running.\n"
                "Install: https://docs.docker.com/get-docker/"
            )
        
        # Ensure base path exists
        os.makedirs(self.base_path, exist_ok=True)
        self.logger.info(f"SandboxManager initialized with base_path={self.base_path}")
    
    def create_sandbox(self, task_id: str, environment_spec: Optional[Dict[str, Any]] = None) -> str:
        """
        Create an isolated Docker container for task execution.
        
        Args:
            task_id: Unique identifier for the task
            environment_spec: Environment configuration including:
                - dockerfile_path: Path to task's Dockerfile
                - docker_compose_path: Path to task's docker-compose.yaml
                - task_path: Path to task directory
                - working_directory: Working directory in container (default: /app)
            
        Returns:
            Sandbox ID for referencing this environment
        """
        if not environment_spec:
            raise ValueError("Environment specification is required")
        
        sandbox_id = f"sandbox_{task_id}_{int(time.time())}".replace("/", "_").replace("-", "_")
        sandbox_path = Path(self.base_path) / sandbox_id
        sandbox_path.mkdir(parents=True, exist_ok=True)
        
        # Get Docker configuration
        docker_compose_path = environment_spec.get("docker_compose_path")
        dockerfile_path = environment_spec.get("dockerfile_path")
        task_path = environment_spec.get("task_path")
        
        self.logger.info(f"Creating Docker sandbox {sandbox_id}")
        self.logger.info(f"  docker-compose: {docker_compose_path}")
        self.logger.info(f"  dockerfile: {dockerfile_path}")
        self.logger.info(f"  task_path: {task_path}")
        
        try:
            # Prefer docker-compose (Terminal Bench's approach)
            if docker_compose_path and Path(docker_compose_path).exists():
                self.logger.info(f"Using docker-compose: {docker_compose_path}")
                container = self._create_with_docker_compose(
                    sandbox_id, docker_compose_path, task_path, sandbox_path
                )
            # Use Dockerfile if available
            elif dockerfile_path and Path(dockerfile_path).exists():
                self.logger.info(f"Using Dockerfile: {dockerfile_path}")
                container = self._create_with_dockerfile(
                    sandbox_id, task_id, dockerfile_path, sandbox_path
                )
            # No Docker configuration - use base image as last resort
            else:
                self.logger.warning(
                    f"No Docker configuration found for task {task_id}. "
                    f"Using base Python image. This may not have required dependencies."
                )
                container = self._create_with_base_image(sandbox_id, sandbox_path)
            
            # Store sandbox information
            self.active_sandboxes[sandbox_id] = {
                "path": str(sandbox_path),
                "container": container,
                "container_id": container.id,
                "container_name": container.name,
                "environment_spec": environment_spec,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "command_history": [],
                "file_snapshots": [],
                "docker_compose_path": docker_compose_path,
                "image_name": f"tbench_{sandbox_id}_image",  # Track image for cleanup
            }
            
            self.logger.info(f"Created Docker sandbox {sandbox_id} with container {container.name}")
            return sandbox_id
            
        except Exception as e:
            self.logger.error(f"Failed to create sandbox {sandbox_id}: {e}")
            raise
    
    def _create_with_docker_compose(self, sandbox_id: str, docker_compose_path: str,
                                    task_path: Optional[str], sandbox_path: Path) -> Container:
        """
        Create container using docker-compose (Terminal Bench's approach).
        Raises RuntimeError if docker-compose fails.
        """
        compose_path = Path(docker_compose_path)
        
        if not compose_path.exists():
            raise RuntimeError(f"docker-compose.yaml not found: {compose_path}")
        
        compose_dir = compose_path.parent
        container_name = f"tbench_{sandbox_id}_client"
        image_name = f"tbench_{sandbox_id}_image"
        
        # Set up environment variables for docker-compose
        env = os.environ.copy()
        env.update({
            "T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME": container_name,
            "T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME": image_name,
            "T_BENCH_TASK_LOGS_PATH": str(sandbox_path.absolute()),
            "T_BENCH_CONTAINER_LOGS_PATH": self.CONTAINER_LOGS_PATH,
            "T_BENCH_TASK_AGENT_LOGS_PATH": str(sandbox_path.absolute()),
            "T_BENCH_CONTAINER_AGENT_LOGS_PATH": self.CONTAINER_AGENT_LOGS_PATH,
            "T_BENCH_TEST_DIR": self.CONTAINER_TEST_DIR,
        })
        
        try:
            # Build with docker-compose
            self.logger.info(f"Building with docker-compose: {compose_path}")
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "build"],
                cwd=compose_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                error_msg = f"Build failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                self.logger.error(error_msg)
                raise RuntimeError(f"docker-compose build failed: {result.stderr}")
            
            # Start with docker-compose
            self.logger.info(f"Starting container with docker-compose")
            
            # Clean up any existing container with the same name first (using Docker API, fast)
            try:
                existing_container = self.docker_client.containers.get(container_name)
                self.logger.info(f"Removing existing container: {container_name}")
                existing_container.remove(force=True)
                self.logger.info(f"Existing container removed")
            except docker.errors.NotFound:
                # No existing container, which is fine
                pass
            except Exception as e:
                self.logger.warning(f"Failed to remove existing container (continuing anyway): {e}")
            
            # Start the container
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "up", "-d"],
                cwd=compose_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode != 0:
                error_msg = f"Start failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                self.logger.error(error_msg)
                raise RuntimeError(f"docker-compose up failed: {result.stderr}")
            
            # Get the container
            try:
                container = self.docker_client.containers.get(container_name)
            except docker.errors.NotFound:
                raise RuntimeError(
                    f"Container {container_name} not found after docker-compose up. "
                    f"Check docker-compose.yaml configuration."
                )
            
            self.logger.info(f"Docker compose container started: {container_name}")
            return container
            
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"docker-compose timed out: {e}")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if hasattr(e, 'stderr') and e.stderr else str(e)
            raise RuntimeError(f"docker-compose command failed: {error_msg}")
        except Exception as e:
            raise RuntimeError(f"docker-compose setup failed: {e}")
    
    def _create_with_dockerfile(self, sandbox_id: str, task_id: str,
                                dockerfile_path: str, sandbox_path: Path) -> Container:
        """
        Create container from Dockerfile.
        Raises RuntimeError if Docker build or run fails.
        """
        dockerfile = Path(dockerfile_path)
        
        if not dockerfile.exists():
            raise RuntimeError(f"Dockerfile not found: {dockerfile}")
        
        build_context = dockerfile.parent
        image_name = f"tbench_{task_id}:{sandbox_id}"
        
        try:
            self.logger.info(f"Building Docker image from {dockerfile}")
            image, build_logs = self.docker_client.images.build(
                path=str(build_context),
                dockerfile=dockerfile.name,
                tag=image_name,
                rm=True
            )
            
            # Log build output
            for log in build_logs:
                if 'stream' in log:
                    self.logger.debug(log['stream'].strip())
            
        except docker.errors.BuildError as e:
            self.logger.error(f"Docker build failed: {e}")
            raise RuntimeError(f"Docker build failed for {dockerfile}: {e}")
        except Exception as e:
            raise RuntimeError(f"Docker build error: {e}")
        
        try:
            # Create and start container
            container_name = f"tbench_{sandbox_id}"
            container = self.docker_client.containers.run(
                image_name,
                command=["sh", "-c", "sleep infinity"],
                name=container_name,
                detach=True,
                working_dir=self.CONTAINER_WORK_DIR,
                volumes={
                    str(sandbox_path.absolute()): {
                        "bind": self.CONTAINER_LOGS_PATH,
                        "mode": "rw"
                    }
                },
                remove=False
            )
            
        except docker.errors.ContainerError as e:
            raise RuntimeError(f"Container failed to start: {e}")
        except docker.errors.ImageNotFound as e:
            raise RuntimeError(f"Image not found after build: {e}")
        except docker.errors.APIError as e:
            raise RuntimeError(f"Docker API error: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to create container: {e}")
        
        try:
            # Ensure /app directory exists
            exec_result = container.exec_run(f"mkdir -p {self.CONTAINER_WORK_DIR}")
            if exec_result.exit_code != 0:
                self.logger.warning(f"Failed to create {self.CONTAINER_WORK_DIR}: {exec_result.output}")
        except Exception as e:
            self.logger.warning(f"Could not create working directory: {e}")
        
        self.logger.info(f"Container created from Dockerfile: {container_name}")
        return container
    
    def _create_with_base_image(self, sandbox_id: str, sandbox_path: Path) -> Container:
        """
        Create container from base Python image.
        This is a last resort when no Docker configuration is provided.
        Raises RuntimeError if Docker operations fail.
        """
        image_name = "python:3.12-slim"
        
        try:
            self.logger.info(f"Pulling base image: {image_name}")
            self.docker_client.images.pull(image_name)
        except docker.errors.APIError as e:
            raise RuntimeError(f"Failed to pull base image {image_name}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error pulling base image: {e}")
        
        try:
            container_name = f"tbench_{sandbox_id}"
            container = self.docker_client.containers.run(
                image_name,
                command=["sh", "-c", "sleep infinity"],
                name=container_name,
                detach=True,
                working_dir=self.CONTAINER_WORK_DIR,
                volumes={
                    str(sandbox_path.absolute()): {
                        "bind": self.CONTAINER_LOGS_PATH,
                        "mode": "rw"
                    }
                },
                remove=False
            )
        except docker.errors.ContainerError as e:
            raise RuntimeError(f"Container failed to start: {e}")
        except docker.errors.ImageNotFound as e:
            raise RuntimeError(f"Base image not found: {e}")
        except docker.errors.APIError as e:
            raise RuntimeError(f"Docker API error: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to create container from base image: {e}")
        
        try:
            # Ensure /app directory exists
            exec_result = container.exec_run(f"mkdir -p {self.CONTAINER_WORK_DIR}")
            if exec_result.exit_code != 0:
                self.logger.warning(f"Failed to create {self.CONTAINER_WORK_DIR}")
        except Exception as e:
            self.logger.warning(f"Could not create working directory: {e}")
        
        self.logger.info(f"Container created from base image: {container_name}")
        return container
    
    def execute_command(self, sandbox_id: str, command: str, timeout: int = 30) -> CommandResult:
        """
        Execute a command inside the Docker container.
        
        Args:
            sandbox_id: ID of the sandbox container
            command: Command to execute
            timeout: Maximum execution time in seconds (not currently enforced)
            
        Returns:
            CommandResult with execution details
            
        Raises:
            ValueError: If sandbox not found
            RuntimeError: If container is not available or exec fails
        """
        if sandbox_id not in self.active_sandboxes:
            raise ValueError(f"Sandbox {sandbox_id} not found in active sandboxes")
        
        sandbox_info = self.active_sandboxes[sandbox_id]
        container = sandbox_info.get("container")
        
        if not container:
            raise RuntimeError(f"No container found for sandbox {sandbox_id}")
        
        start_time = time.time()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        try:
            self.logger.info(f"Executing in {sandbox_id}: {command}")
            
            # Execute command in container (Terminal Bench style)
            exec_result = container.exec_run(
                cmd=["sh", "-c", command],
                workdir=self.CONTAINER_WORK_DIR,
                demux=True
            )
            
            execution_time = time.time() - start_time
            
            # Decode output
            stdout = exec_result.output[0].decode('utf-8') if exec_result.output[0] else ""
            stderr = exec_result.output[1].decode('utf-8') if exec_result.output[1] else ""
            
            command_result = CommandResult(
                command=command,
                stdout=stdout,
                stderr=stderr,
                returncode=exec_result.exit_code,
                execution_time=execution_time,
                timestamp=timestamp,
                success=exec_result.exit_code == 0
            )
            
            # Store command in history
            sandbox_info["command_history"].append(command_result)
            
            self.logger.info(
                f"Command completed in {execution_time:.2f}s with exit code {exec_result.exit_code}"
            )
            return command_result
            
        except docker.errors.APIError as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Docker API error during command execution: {e}")
            raise RuntimeError(f"Docker command execution failed: {e}")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Command execution failed: {e}")
            raise RuntimeError(f"Failed to execute command in container: {e}")
    
    def capture_state(self, sandbox_id: str) -> SandboxState:
        """
        Capture current state of the Docker container.
        
        Args:
            sandbox_id: ID of the sandbox to capture
            
        Returns:
            SandboxState with current container state
        """
        if sandbox_id not in self.active_sandboxes:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        
        sandbox_info = self.active_sandboxes[sandbox_id]
        
        # Capture container filesystem state
        file_snapshot = self._capture_container_filesystem(sandbox_info)
        
        # Store snapshot
        sandbox_info["file_snapshots"].append(file_snapshot)
        
        return SandboxState(
            sandbox_id=sandbox_id,
            working_directory=self.CONTAINER_WORK_DIR,
            environment_vars={},
            file_system_snapshot=file_snapshot,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def _capture_container_filesystem(self, sandbox_info: Dict[str, Any]) -> Dict[str, Any]:
        """Capture filesystem state from Docker container."""
        snapshot = {
            "files": {},
            "directories": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            container = sandbox_info["container"]
            
            # List files in /app directory
            exec_result = container.exec_run(
                cmd=["sh", "-c", f"find {self.CONTAINER_WORK_DIR} -type f 2>/dev/null || true"],
                workdir=self.CONTAINER_WORK_DIR
            )
            
            if exec_result.exit_code == 0:
                file_list = exec_result.output.decode('utf-8').strip().split('\n')
                
                for file_path in file_list:
                    if file_path:
                        # Read file content
                        read_result = container.exec_run(
                            cmd=["cat", file_path],
                            workdir="/"
                        )
                        if read_result.exit_code == 0:
                            content = read_result.output.decode('utf-8', errors='ignore')
                            snapshot["files"][file_path] = content
            
            # List directories
            exec_result = container.exec_run(
                cmd=["sh", "-c", f"find {self.CONTAINER_WORK_DIR} -type d 2>/dev/null || true"],
                workdir=self.CONTAINER_WORK_DIR
            )
            
            if exec_result.exit_code == 0:
                dir_list = exec_result.output.decode('utf-8').strip().split('\n')
                snapshot["directories"] = [d for d in dir_list if d]
                
        except Exception as e:
            self.logger.error(f"Failed to capture container filesystem: {e}")
        
        return snapshot
    
    def destroy_sandbox(self, sandbox_id: str, cleanup_images: bool = True):
        """
        Stop and remove Docker container, images, and clean up resources.
        
        Args:
            sandbox_id: ID of the sandbox to destroy
            cleanup_images: Whether to remove Docker images (default: True)
        """
        if sandbox_id not in self.active_sandboxes:
            self.logger.warning(f"Sandbox {sandbox_id} not found for destruction")
            return
        
        sandbox_info = self.active_sandboxes[sandbox_id]
        image_name = sandbox_info.get("image_name")
        
        try:
            # If using docker-compose, use docker-compose down
            if sandbox_info.get("docker_compose_path"):
                self._cleanup_docker_compose(sandbox_info, cleanup_images=cleanup_images)
            
            # Stop and remove container
            container = sandbox_info.get("container")
            if container:
                try:
                    self.logger.info(f"Stopping container {container.name}")
                    container.stop(timeout=10)
                    container.remove()
                    self.logger.info(f"Container {container.name} stopped and removed")
                except Exception as e:
                    self.logger.warning(f"Failed to stop/remove container: {e}")
            
            # Remove Docker image if requested
            if cleanup_images and image_name:
                self._cleanup_docker_image(image_name)
            
            # Remove from active sandboxes
            del self.active_sandboxes[sandbox_id]
            
            self.logger.info(f"Destroyed sandbox {sandbox_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to destroy sandbox {sandbox_id}: {e}")
    
    def _cleanup_docker_compose(self, sandbox_info: Dict[str, Any], cleanup_images: bool = True):
        """Clean up docker-compose services and optionally remove images."""
        try:
            compose_path = sandbox_info["docker_compose_path"]
            compose_dir = Path(compose_path).parent
            
            # Get environment variables
            env = os.environ.copy()
            sandbox_id = sandbox_info.get("container_name", "").replace("tbench_", "").replace("_client", "")
            image_name = f"tbench_{sandbox_id}_image"
            env.update({
                "T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME": sandbox_info.get("container_name", ""),
                "T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME": image_name,
            })
            
            # Use --volumes to also remove volumes, and --rmi to remove images
            compose_cmd = ["docker", "compose", "-f", compose_path, "down", "-v"]
            if cleanup_images:
                compose_cmd.append("--rmi")
                compose_cmd.append("all")  # Remove all images used by services
            
            self.logger.info(f"Stopping docker-compose services (cleanup_images={cleanup_images})")
            result = subprocess.run(
                compose_cmd,
                cwd=compose_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                self.logger.info(f"Docker-compose services cleaned up")
            else:
                self.logger.warning(f"Docker-compose down had issues: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Docker-compose cleanup timed out")
        except Exception as e:
            self.logger.warning(f"Error during docker-compose cleanup: {e}")
    
    def _cleanup_docker_image(self, image_name: str):
        """Remove Docker image by name."""
        try:
            self.logger.info(f"Removing Docker image: {image_name}")
            
            # Try to remove the image
            try:
                image = self.docker_client.images.get(image_name)
                self.docker_client.images.remove(image.id, force=True)
                self.logger.info(f"Removed Docker image: {image_name}")
            except docker.errors.ImageNotFound:
                self.logger.debug(f"Image {image_name} not found (already removed or never created)")
            except docker.errors.APIError as e:
                self.logger.warning(f"Could not remove image {image_name}: {e}")
            
            # Also try to prune dangling images created during this task
            try:
                pruned = self.docker_client.images.prune(filters={"dangling": True})
                if pruned.get("ImagesDeleted"):
                    deleted_count = len(pruned["ImagesDeleted"])
                    self.logger.info(f"Pruned {deleted_count} dangling images")
            except Exception as e:
                self.logger.debug(f"Image prune failed: {e}")
                
        except Exception as e:
            self.logger.warning(f"Error cleaning up Docker image {image_name}: {e}")
    
    def get_sandbox_info(self, sandbox_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a sandbox."""
        return self.active_sandboxes.get(sandbox_id)
    
    def list_sandboxes(self) -> List[str]:
        """List all active sandbox IDs."""
        return list(self.active_sandboxes.keys())
    
    def cleanup_all(self, cleanup_images: bool = True):
        """
        Clean up all Docker containers and resources.
        
        Args:
            cleanup_images: Whether to remove Docker images (default: True)
        """
        self.logger.info("Cleaning up all sandbox resources")
        
        for sandbox_id in list(self.active_sandboxes.keys()):
            self.destroy_sandbox(sandbox_id, cleanup_images=cleanup_images)
        
        if cleanup_images:
            # Final prune of all unused images and build cache
            try:
                self.logger.info("Pruning unused Docker resources...")
                self.docker_client.images.prune(filters={"dangling": True})
                # Also prune build cache
                self.docker_client.api.prune_builds()
                self.logger.info("Pruned unused Docker resources")
            except Exception as e:
                self.logger.debug(f"Final cleanup prune failed: {e}")
        
        self.logger.info("All sandbox resources cleaned up")
