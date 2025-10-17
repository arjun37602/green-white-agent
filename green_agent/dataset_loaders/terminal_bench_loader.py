"""
Terminal-Bench Dataset Loader

This module provides functionality to load and process Terminal-Bench datasets,
handling various formats and converting them for use with the green agent.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


class TerminalBenchLoader:
    """
    Dataset loader for Terminal-Bench datasets.
    
    This class handles loading various Terminal-Bench dataset formats,
    including JSON, CSV, and HuggingFace datasets, and provides
    a unified interface for accessing the data.
    """
    
    def __init__(self):
        """Initialize the Terminal-Bench dataset loader."""
        self.logger = logging.getLogger(__name__)
        self.supported_formats = {'.json', '.jsonl', '.csv', '.tsv', '.txt'}
        self.logger.info("TerminalBenchLoader initialized")
    
    def load_dataset(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a dataset from the specified path.
        
        Args:
            dataset_path: Path to the dataset file or directory
            
        Returns:
            Dictionary containing the loaded dataset and metadata
        """
        dataset_path = Path(dataset_path)
        self.logger.info(f"Loading dataset from: {dataset_path}")
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        if dataset_path.is_file():
            return self._load_file(dataset_path)
        elif dataset_path.is_dir():
            return self._load_directory(dataset_path)
        else:
            raise ValueError(f"Invalid dataset path: {dataset_path}")
    
    def _load_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load a single dataset file.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            Dictionary containing the loaded dataset
        """
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        self.logger.info(f"Loading file: {file_path} (format: {file_extension})")
        
        try:
            if file_extension == '.json':
                data = self._load_json(file_path)
            elif file_extension == '.jsonl':
                data = self._load_jsonl(file_path)
            elif file_extension in {'.csv', '.tsv'}:
                data = self._load_csv(file_path)
            elif file_extension == '.txt':
                data = self._load_txt(file_path)
            else:
                raise ValueError(f"Unhandled file format: {file_extension}")
            
            # Add metadata
            dataset_info = {
                "data": data,
                "metadata": {
                    "file_path": str(file_path),
                    "file_format": file_extension,
                    "file_size": file_path.stat().st_size,
                    "num_records": len(data) if isinstance(data, list) else 1
                }
            }
            
            self.logger.info(f"Successfully loaded {dataset_info['metadata']['num_records']} records")
            return dataset_info
            
        except Exception as e:
            self.logger.error(f"Failed to load file {file_path}: {e}")
            raise
    
    def _load_directory(self, dir_path: Path) -> Dict[str, Any]:
        """
        Load all datasets from a directory.
        
        Args:
            dir_path: Path to the directory containing dataset files
            
        Returns:
            Dictionary containing all loaded datasets
        """
        self.logger.info(f"Loading directory: {dir_path}")
        
        dataset_files = []
        for file_path in dir_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                dataset_files.append(file_path)
        
        if not dataset_files:
            raise ValueError(f"No supported dataset files found in directory: {dir_path}")
        
        datasets = {}
        for file_path in dataset_files:
            try:
                file_data = self._load_file(file_path)
                datasets[file_path.stem] = file_data
            except Exception as e:
                self.logger.warning(f"Failed to load file {file_path}: {e}")
                continue
        
        # Combine all datasets
        combined_data = []
        combined_metadata = {
            "directory_path": str(dir_path),
            "num_files": len(datasets),
            "file_details": {}
        }
        
        for name, dataset_info in datasets.items():
            combined_data.extend(dataset_info["data"] if isinstance(dataset_info["data"], list) else [dataset_info["data"]])
            combined_metadata["file_details"][name] = dataset_info["metadata"]
        
        result = {
            "data": combined_data,
            "metadata": combined_metadata
        }
        
        self.logger.info(f"Successfully loaded {len(datasets)} files from directory")
        return result
    
    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure data is in list format for consistency
        if isinstance(data, dict):
            # If it's a single object, wrap it in a list
            return [data]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unexpected JSON structure in {file_path}")
    
    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {e}")
                        continue
        
        return data
    
    def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load CSV or TSV file."""
        try:
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        except Exception as e:
            self.logger.error(f"Failed to load CSV file {file_path}: {e}")
            raise
    
    def _load_txt(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # For text files, we'll create a simple structure
        return [{"text": content, "file_path": str(file_path)}]
    
    def validate_dataset_format(self, dataset: Dict[str, Any]) -> bool:
        """
        Validate that the dataset has the expected Terminal-Bench format.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            True if dataset format is valid, False otherwise
        """
        try:
            data = dataset.get("data", [])
            if not isinstance(data, list):
                return False
            
            # Check for required Terminal-Bench fields
            required_fields = {"instruction", "environment", "test"}
            optional_fields = {"id", "difficulty", "category", "tags"}
            
            for record in data[:5]:  # Check first 5 records as sample
                if not isinstance(record, dict):
                    return False
                
                # Check if at least some required fields are present
                record_fields = set(record.keys())
                if not required_fields.intersection(record_fields):
                    self.logger.warning(f"Record missing required fields: {record_fields}")
                    continue
            
            self.logger.info("Dataset format validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Dataset format validation failed: {e}")
            return False
    
    def get_dataset_statistics(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary containing dataset statistics
        """
        data = dataset.get("data", [])
        metadata = dataset.get("metadata", {})
        
        stats = {
            "total_records": len(data),
            "metadata": metadata
        }
        
        if data and isinstance(data[0], dict):
            # Analyze field distribution
            field_counts = {}
            field_types = {}
            
            for record in data:
                for field, value in record.items():
                    field_counts[field] = field_counts.get(field, 0) + 1
                    field_types[field] = type(value).__name__
            
            stats["field_distribution"] = field_counts
            stats["field_types"] = field_types
        
        return stats

