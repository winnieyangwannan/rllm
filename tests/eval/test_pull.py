"""Tests for _pull.py: field_map, hf_config, and transform support."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rllm.experimental.cli._pull import _load_transform, _remap_fields


class TestRemapFields:
    def test_basic_remap(self):
        row = {"prompt": "hello", "answer": "world", "extra": "keep"}
        field_map = {"prompt": "question", "answer": "ground_truth"}
        result = _remap_fields(row, field_map)
        assert result["question"] == "hello"
        assert result["ground_truth"] == "world"
        assert result["extra"] == "keep"
        assert "prompt" not in result
        assert "answer" not in result

    def test_missing_source_field(self):
        row = {"prompt": "hello"}
        field_map = {"prompt": "question", "missing_field": "target"}
        result = _remap_fields(row, field_map)
        assert result["question"] == "hello"
        assert "target" not in result

    def test_empty_field_map(self):
        row = {"a": 1, "b": 2}
        result = _remap_fields(row, {})
        assert result == {"a": 1, "b": 2}


class TestLoadTransform:
    def test_load_valid_transform(self):
        fn = _load_transform("rllm.data.transforms:gpqa_diamond_transform")
        assert callable(fn)

    def test_load_invalid_module(self):
        with pytest.raises(ModuleNotFoundError):
            _load_transform("nonexistent.module:func")

    def test_load_invalid_function(self):
        with pytest.raises(AttributeError):
            _load_transform("rllm.data.transforms:nonexistent_function")


class TestPullDatasetWithTransform:
    @patch("datasets.load_dataset")
    @patch("rllm.data.DatasetRegistry.register_dataset")
    def test_pull_with_field_map(self, mock_register, mock_load_dataset):
        from rllm.experimental.cli._pull import pull_dataset

        # Mock HF dataset
        mock_hf = MagicMock()
        mock_hf.__iter__ = MagicMock(return_value=iter([{"prompt": "hello", "answer": "world"}]))
        mock_hf.__len__ = MagicMock(return_value=1)
        mock_load_dataset.return_value = mock_hf

        catalog_entry = {
            "source": "test/dataset",
            "splits": ["test"],
            "field_map": {"prompt": "question", "answer": "ground_truth"},
        }

        pull_dataset("test_ds", catalog_entry)

        # Check register was called with remapped data
        call_args = mock_register.call_args
        data = call_args[1]["data"] if "data" in call_args[1] else call_args[0][1]
        assert data[0]["question"] == "hello"
        assert data[0]["ground_truth"] == "world"

    @patch("datasets.load_dataset")
    @patch("rllm.data.DatasetRegistry.register_dataset")
    def test_pull_with_hf_config(self, mock_register, mock_load_dataset):
        from rllm.experimental.cli._pull import pull_dataset

        mock_hf = MagicMock()
        mock_hf.__iter__ = MagicMock(return_value=iter([]))
        mock_hf.__len__ = MagicMock(return_value=0)
        mock_load_dataset.return_value = mock_hf

        catalog_entry = {
            "source": "test/dataset",
            "splits": ["train"],
            "hf_config": "diamond",
        }

        pull_dataset("test_ds", catalog_entry)

        # Check that load_dataset was called with name=diamond
        mock_load_dataset.assert_called_once_with(path="test/dataset", split="train", name="diamond")

    @patch("datasets.load_dataset")
    @patch("rllm.data.DatasetRegistry.register_dataset")
    def test_pull_with_transform(self, mock_register, mock_load_dataset):
        from rllm.experimental.cli._pull import pull_dataset

        mock_hf = MagicMock()
        mock_hf.__iter__ = MagicMock(return_value=iter([
            {
                "Question": "Test Q",
                "Correct Answer": "Right",
                "Incorrect Answer 1": "Wrong1",
                "Incorrect Answer 2": "Wrong2",
                "Incorrect Answer 3": "Wrong3",
            }
        ]))
        mock_hf.__len__ = MagicMock(return_value=1)
        mock_load_dataset.return_value = mock_hf

        catalog_entry = {
            "source": "test/gpqa",
            "splits": ["train"],
            "transform": "rllm.data.transforms:gpqa_diamond_transform",
        }

        pull_dataset("gpqa_test", catalog_entry)

        call_args = mock_register.call_args
        data = call_args[1]["data"] if "data" in call_args[1] else call_args[0][1]
        assert data[0]["question"] == "Test Q"
        assert len(data[0]["choices"]) == 4
        assert "Right" in data[0]["choices"]
        assert data[0]["data_source"] == "gpqa_diamond"
