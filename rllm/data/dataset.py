"""Unified dataset interface for loading and processing datasets.

This module provides a flexible dataset loading system that supports:
1. Loading datasets by name (registered or HuggingFace)
2. Custom postprocessing functions
3. Fallback from local files to HuggingFace
4. Dataset registration for custom datasets
5. Persistent dataset registry across script executions
"""

import json
import os
import warnings
import importlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from datasets import load_dataset as hf_load_dataset
from rllm.data.dataset_types import TrainDataset, TestDataset
from rllm.data.utils import load_dataset as local_load_dataset


class DatasetRegistry:
    """Registry for custom datasets and their postprocessing functions with persistence."""
    
    _registry: Dict[str, Dict[str, Any]] = {}
    _config_file: Optional[str] = None
    
    @classmethod
    def set_config_file(cls, config_path: str):
        """Set the path for persistent registry configuration.
        
        Args:
            config_path: Path to save/load registry configuration
        """
        cls._config_file = config_path
        cls._load_config()
    
    @classmethod
    def _get_default_config_path(cls) -> str:
        """Get default configuration file path."""
        # Save in the rllm data directory
        import rllm
        rllm_path = Path(rllm.__file__).parent
        config_dir = rllm_path / "data" / "configs"
        config_dir.mkdir(exist_ok=True)
        return str(config_dir / "dataset_registry.json")
    
    @classmethod
    def _load_config(cls):
        """Load registry configuration from file."""
        config_path = cls._config_file or cls._get_default_config_path()
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Load registered datasets
                for dataset_name, dataset_config in config.items():
                    cls._registry[dataset_name] = cls._deserialize_config(dataset_config)
                    
            except Exception as e:
                warnings.warn(f"Failed to load dataset registry config: {e}")
    
    @classmethod
    def _serialize_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize configuration for JSON storage."""
        serialized = config.copy()
        
        # Handle postprocess function serialization
        if 'postprocess_fn' in serialized and serialized['postprocess_fn'] is not None:
            fn = serialized['postprocess_fn']
            if hasattr(fn, '__module__') and hasattr(fn, '__name__'):
                serialized['postprocess_fn_module'] = fn.__module__
                serialized['postprocess_fn_name'] = fn.__name__
            del serialized['postprocess_fn']
        
        return serialized
    
    @classmethod
    def _deserialize_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize configuration from JSON storage."""
        deserialized = config.copy()
        
        # Handle postprocess function deserialization
        if 'postprocess_fn_module' in deserialized and 'postprocess_fn_name' in deserialized:
            try:
                module = importlib.import_module(deserialized['postprocess_fn_module'])
                fn = getattr(module, deserialized['postprocess_fn_name'])
                deserialized['postprocess_fn'] = fn
            except (ImportError, AttributeError) as e:
                warnings.warn(f"Failed to import postprocess function: {e}")
                deserialized['postprocess_fn'] = None
            
            # Clean up serialization keys
            del deserialized['postprocess_fn_module']
            del deserialized['postprocess_fn_name']
        else:
            deserialized['postprocess_fn'] = None
        
        return deserialized
    
    @classmethod
    def save_config(cls):
        """Save current registry configuration to file."""
        config_path = cls._config_file or cls._get_default_config_path()
        
        # Serialize the registry
        serialized_registry = {}
        for dataset_name, dataset_config in cls._registry.items():
            serialized_registry[dataset_name] = cls._serialize_config(dataset_config)
        
        # Save to file
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(serialized_registry, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def register_dataset(
        cls,
        dataset_name: str,
        source_type: str = "hf",
        hf_dataset_name: Optional[str] = None,
        local_path: Optional[str] = None,
        dataset_enum: Optional[str] = None,
        postprocess_fn: Optional[Callable] = None,
        trust_remote_code: bool = False,
        save_to_config: bool = True,
        **kwargs
    ):
        """Register a dataset with flexible source types and optional persistence.
        
        Args:
            dataset_name: Name to register the dataset as
            source_type: Type of data source ("hf", "local", "enum", "custom")
            hf_dataset_name: HuggingFace dataset name (for "hf" source_type)
            local_path: Local file path (for "local" source_type)
            dataset_enum: Dataset enum string (for "enum" source_type)
            postprocess_fn: Function to postprocess dataset examples
            trust_remote_code: Whether to trust remote code for HF datasets
            save_to_config: Whether to save registration to persistent config
            **kwargs: Additional arguments for dataset loading
            
        Returns:
            If used as decorator, returns the postprocess_fn
        """
        def decorator(fn):
            config = {
                'source_type': source_type,
                'trust_remote_code': trust_remote_code,
                'postprocess_fn': fn,
                **kwargs
            }
            
            # Add source-specific configuration
            if source_type == "hf":
                config['hf_dataset_name'] = hf_dataset_name or dataset_name.lower()
            elif source_type == "local":
                config['local_path'] = local_path
            elif source_type == "enum":
                config['dataset_enum'] = dataset_enum
            
            cls._registry[dataset_name] = config
            
            if save_to_config:
                cls.save_config()
            
            return fn
            
        # If called with a function (used as decorator with parentheses)
        if postprocess_fn is not None:
            return decorator(postprocess_fn)
        
        # If used directly (not as decorator)
        config = {
            'source_type': source_type,
            'trust_remote_code': trust_remote_code,
            'postprocess_fn': None,
            **kwargs
        }
        
        # Add source-specific configuration
        if source_type == "hf":
            config['hf_dataset_name'] = hf_dataset_name or dataset_name.lower()
        elif source_type == "local":
            config['local_path'] = local_path
        elif source_type == "enum":
            config['dataset_enum'] = dataset_enum
        
        cls._registry[dataset_name] = config
        
        if save_to_config:
            cls.save_config()
        
        # Return decorator for use as @DatasetRegistry.register_dataset(...)
        return decorator
    
    @classmethod
    def get_dataset_info(cls, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get registered dataset information."""
        # Load config if not already loaded
        if not cls._registry and not cls._config_file:
            cls._load_config()
        return cls._registry.get(dataset_name)
    
    @classmethod
    def list_registered_datasets(cls) -> List[str]:
        """List all registered dataset names."""
        # Load config if not already loaded
        if not cls._registry and not cls._config_file:
            cls._load_config()
        return list(cls._registry.keys())
    
    @classmethod
    def unregister_dataset(cls, dataset_name: str, save_to_config: bool = True):
        """Unregister a dataset.
        
        Args:
            dataset_name: Name of dataset to unregister
            save_to_config: Whether to save changes to persistent config
        """
        if dataset_name in cls._registry:
            del cls._registry[dataset_name]
            if save_to_config:
                cls.save_config()
    
    @classmethod
    def clear_registry(cls, save_to_config: bool = True):
        """Clear all registered datasets.
        
        Args:
            save_to_config: Whether to save changes to persistent config
        """
        cls._registry.clear()
        if save_to_config:
            cls.save_config()


class Dataset:
    """Unified dataset class for loading and processing datasets.
    
    Supports loading from:
    1. Local JSON files (using existing rllm.data.utils.load_dataset)
    2. HuggingFace datasets
    3. Registered custom datasets with multiple source types
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
        """Load a registered dataset based on its source type."""
        source_type = registry_info.get('source_type', 'hf')
        registered_postprocess_fn = registry_info.get('postprocess_fn')
        
        # Use registered postprocess function if no custom one provided
        if self.postprocess_fn is None:
            self.postprocess_fn = registered_postprocess_fn
        
        # Update trust_remote_code from registry if not explicitly set
        if registry_info.get('trust_remote_code', False):
            self.trust_remote_code = True
        
        try:
            if source_type == "hf":
                hf_name = registry_info.get('hf_dataset_name')
                # Merge registered kwargs with instance kwargs
                merged_kwargs = {}
                # Add registered kwargs (excluding our known keys)
                excluded_keys = {'source_type', 'hf_dataset_name', 'postprocess_fn', 'trust_remote_code', 
                               'local_path', 'dataset_enum'}
                for key, value in registry_info.items():
                    if key not in excluded_keys:
                        merged_kwargs[key] = value
                # Override with instance kwargs
                merged_kwargs.update(self.hf_kwargs)
                
                # Temporarily store original hf_kwargs and update
                original_hf_kwargs = self.hf_kwargs
                self.hf_kwargs = merged_kwargs
                try:
                    result = self._load_hf_dataset(hf_name)
                finally:
                    # Restore original hf_kwargs
                    self.hf_kwargs = original_hf_kwargs
                return result
            
            elif source_type == "local":
                local_path = registry_info.get('local_path')
                if not local_path:
                    raise ValueError(f"No local_path specified for local dataset {self.dataset_name_str}")
                return self._load_local_file(local_path)
            
            elif source_type == "enum":
                dataset_enum_str = registry_info.get('dataset_enum')
                if not dataset_enum_str:
                    raise ValueError(f"No dataset_enum specified for enum dataset {self.dataset_name_str}")
                dataset_enum = self._parse_dataset_enum(dataset_enum_str)
                return self._load_enum_dataset(dataset_enum)
            
            else:
                raise ValueError(f"Unsupported source_type: {source_type}")
                
        except Exception as e:
            # If loading fails and we have an enum as fallback, try local
            if hasattr(self.dataset_name, 'value'):
                try:
                    return self._load_local_dataset()
                except:
                    pass
            raise ValueError(f"Failed to load registered dataset {self.dataset_name_str}: {e}")
    
    def _load_local_file(self, local_path: str) -> List[Dict[str, Any]]:
        """Load dataset from a local file."""
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        try:
            if local_path.endswith('.json'):
                with open(local_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif local_path.endswith('.jsonl'):
                data = []
                with open(local_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
            else:
                raise ValueError(f"Unsupported file format: {local_path}")
            
            return self._apply_postprocessing(data)
            
        except Exception as e:
            raise ValueError(f"Failed to load local file {local_path}: {e}")
    
    def _parse_dataset_enum(self, dataset_enum_str: str):
        """Parse dataset enum string like 'TrainDataset.Code.APPS'."""
        try:
            parts = dataset_enum_str.split('.')
            if len(parts) == 3:
                main_class, sub_class, value = parts
                if main_class == "TrainDataset":
                    return getattr(getattr(TrainDataset, sub_class), value)
                elif main_class == "TestDataset":
                    return getattr(getattr(TestDataset, sub_class), value)
            raise ValueError(f"Invalid dataset enum format: {dataset_enum_str}")
        except Exception as e:
            raise ValueError(f"Failed to parse dataset enum {dataset_enum_str}: {e}")
    
    def _load_enum_dataset(self, dataset_enum) -> List[Dict[str, Any]]:
        """Load dataset using the enum (falls back to existing utils)."""
        try:
            data = local_load_dataset(dataset_enum)
            return self._apply_postprocessing(data)
        except Exception as e:
            raise ValueError(f"Failed to load enum dataset: {e}")
    
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