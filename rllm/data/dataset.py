"""Unified dataset interface for loading and processing datasets.

This module provides a flexible dataset loading system that supports:
1. Loading datasets by name (registered or HuggingFace)
2. Custom postprocessing functions
3. Fallback from local files to HuggingFace
4. Dataset registration for custom datasets
"""

import json
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

from datasets import load_dataset as hf_load_dataset
from rllm.data.dataset_types import TrainDataset, TestDataset
from rllm.data.utils import load_dataset as local_load_dataset


class DatasetRegistry:
    """Registry for custom datasets and their postprocessing functions."""
    
    _registry: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register_dataset(
        cls,
        dataset_name: str,
        hf_dataset_name: Optional[str] = None,
        postprocess_fn: Optional[Callable] = None
    ):
        """Register a dataset with optional HuggingFace name and postprocessing function.
        
        Can be used as a decorator or called directly.
        
        Args:
            dataset_name: Name to register the dataset as
            hf_dataset_name: HuggingFace dataset name (if different from dataset_name)
            postprocess_fn: Function to postprocess dataset examples
            
        Returns:
            If used as decorator, returns the postprocess_fn
        """
        def decorator(fn):
            cls._registry[dataset_name] = {
                'hf_dataset_name': hf_dataset_name or dataset_name.lower(),
                'postprocess_fn': fn
            }
            return fn
            
        # If called with a function (used as decorator with parentheses)
        if postprocess_fn is not None:
            return decorator(postprocess_fn)
        
        # If used directly (not as decorator)
        if hf_dataset_name is not None or dataset_name:
            cls._registry[dataset_name] = {
                'hf_dataset_name': hf_dataset_name or dataset_name.lower(),
                'postprocess_fn': None
            }
        
        # Return decorator for use as @DatasetRegistry.register_dataset(...)
        return decorator
    
    @classmethod
    def get_dataset_info(cls, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get registered dataset information."""
        return cls._registry.get(dataset_name)
    
    @classmethod
    def list_registered_datasets(cls) -> List[str]:
        """List all registered dataset names."""
        return list(cls._registry.keys())


class Dataset:
    """Unified dataset class for loading and processing datasets.
    
    Supports loading from:
    1. Local JSON files (using existing rllm.data.utils.load_dataset)
    2. HuggingFace datasets
    3. Registered custom datasets
    """
    
    def __init__(
        self,
        dataset_name: Union[str, TrainDataset, TestDataset],
        split: str = "train",
        load_from_hf: bool = False,
        trust_remote_code: bool = False,
        postprocess_fn: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
        **hf_kwargs
    ):
        """Initialize dataset.
        
        Args:
            dataset_name: Dataset name or enum
            split: Dataset split to load
            load_from_hf: Whether to load from HuggingFace
            trust_remote_code: Whether to trust remote code for HF datasets
            postprocess_fn: Custom postprocessing function
            cache_dir: Cache directory for HF datasets
            **hf_kwargs: Additional arguments for HuggingFace load_dataset
        """
        self.dataset_name = dataset_name
        self.split = split
        self.load_from_hf = load_from_hf
        self.trust_remote_code = trust_remote_code
        self.postprocess_fn = postprocess_fn
        self.cache_dir = cache_dir
        self.hf_kwargs = hf_kwargs
        
        # Convert enum to string if needed
        if hasattr(dataset_name, 'value'):
            self.dataset_name_str = dataset_name.value
        else:
            self.dataset_name_str = str(dataset_name)
            
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from appropriate source."""
        
        # Check if dataset is registered
        registry_info = DatasetRegistry.get_dataset_info(self.dataset_name_str)
        if registry_info:
            return self._load_registered_dataset(registry_info)
        
        # Try loading from local files first (if dataset_name is an enum)
        if hasattr(self.dataset_name, 'value') and not self.load_from_hf:
            try:
                return self._load_local_dataset()
            except (ValueError, FileNotFoundError) as e:
                if self.load_from_hf:
                    warnings.warn(f"Local loading failed: {e}. Trying HuggingFace...")
                else:
                    raise
        
        # Load from HuggingFace
        if self.load_from_hf:
            return self._load_hf_dataset()
        else:
            raise ValueError(f"Dataset {self.dataset_name_str} not found and load_from_hf=False")
    
    def _load_local_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset from local JSON files using existing utils."""
        try:
            data = local_load_dataset(self.dataset_name)
            return self._apply_postprocessing(data)
        except Exception as e:
            raise ValueError(f"Failed to load local dataset {self.dataset_name_str}: {e}")
    
    def _load_hf_dataset(self, hf_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load dataset from HuggingFace."""
        hf_dataset_name = hf_name or self.dataset_name_str.lower()
        
        try:
            # Load from HuggingFace
            dataset = hf_load_dataset(
                hf_dataset_name,
                split=self.split,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir,
                **self.hf_kwargs
            )
            
            # Convert to list of dictionaries
            data = [dict(example) for example in dataset]
            return self._apply_postprocessing(data)
            
        except Exception as e:
            raise ValueError(f"Failed to load HuggingFace dataset {hf_dataset_name}: {e}")
    
    def _load_registered_dataset(self, registry_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load a registered dataset."""
        hf_name = registry_info.get('hf_dataset_name')
        registered_postprocess_fn = registry_info.get('postprocess_fn')
        
        # Use registered postprocess function if no custom one provided
        if self.postprocess_fn is None:
            self.postprocess_fn = registered_postprocess_fn
        
        # Try HuggingFace loading
        try:
            return self._load_hf_dataset(hf_name)
        except Exception as e:
            # If HF loading fails and we have an enum, try local
            if hasattr(self.dataset_name, 'value'):
                try:
                    return self._load_local_dataset()
                except:
                    pass
            raise ValueError(f"Failed to load registered dataset {self.dataset_name_str}: {e}")
    
    def _apply_postprocessing(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply postprocessing function to data."""
        if self.postprocess_fn is None:
            return data
        
        processed_data = []
        for idx, example in enumerate(data):
            try:
                processed_example = self.postprocess_fn(example, idx)
                if processed_example is not None:
                    processed_data.append(processed_example)
            except Exception as e:
                warnings.warn(f"Postprocessing failed for example {idx}: {e}")
                continue
        
        return processed_data
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item by index."""
        return self.data[idx]
    
    def __iter__(self):
        """Iterate over dataset."""
        return iter(self.data)
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Return data as list."""
        return self.data.copy()
    
    def save(self, output_path: str):
        """Save dataset to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False) 