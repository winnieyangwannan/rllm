"""Tests for VLM dataset transform functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rllm.data.transforms import (
    babyvision_transform,
    dynamath_transform,
    mathvision_transform,
    mathvista_transform,
    mmmu_pro_transform,
    mmmu_transform,
    vlmsareblind_transform,
    zerobench_sub_transform,
    zerobench_transform,
)


def _mock_image(name: str = "test") -> MagicMock:
    """Create a mock PIL Image object."""
    mock = MagicMock()
    mock.__class__.__name__ = "Image"
    return mock


# ---------------------------------------------------------------------------
# MMMU
# ---------------------------------------------------------------------------


class TestMMMUTransform:
    def test_basic_transform(self):
        img1 = _mock_image()
        img2 = _mock_image()
        row = {
            "question": "What does this diagram show?",
            "image_1": img1,
            "image_2": img2,
            "image_3": None,
            "options": '["Cell division", "DNA replication", "Protein synthesis", "Gene expression"]',
            "answer": "A",
            "subfield": "Biology",
        }
        result = mmmu_transform(row)
        assert result["question"] == "What does this diagram show?"
        assert result["images"] == [img1, img2]
        assert result["choices"] == ["Cell division", "DNA replication", "Protein synthesis", "Gene expression"]
        assert result["ground_truth"] == "A"
        assert result["data_source"] == "mmmu"
        assert result["subject"] == "Biology"

    def test_no_images(self):
        row = {
            "question": "Text only question",
            "options": '["A", "B"]',
            "answer": "B",
            "subfield": "Math",
        }
        result = mmmu_transform(row)
        assert result["images"] == []
        assert result["choices"] == ["A", "B"]

    def test_options_as_list(self):
        row = {
            "question": "Q",
            "options": ["X", "Y", "Z"],
            "answer": "C",
            "subfield": "Test",
        }
        result = mmmu_transform(row)
        assert result["choices"] == ["X", "Y", "Z"]

    def test_malformed_options_string(self):
        row = {
            "question": "Q",
            "options": "not valid json",
            "answer": "A",
            "subfield": "Test",
        }
        result = mmmu_transform(row)
        assert result["choices"] == []


# ---------------------------------------------------------------------------
# MMMU-Pro
# ---------------------------------------------------------------------------


class TestMMMUProTransform:
    def test_basic_transform(self):
        img = _mock_image()
        row = {
            "question": "What is shown?",
            "image_1": img,
            "options": '["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]',
            "answer": "D",
            "subject": "Physics",
        }
        result = mmmu_pro_transform(row)
        assert result["question"] == "What is shown?"
        assert result["images"] == [img]
        assert len(result["choices"]) == 10
        assert result["ground_truth"] == "D"
        assert result["data_source"] == "mmmu_pro"
        assert result["subject"] == "Physics"


# ---------------------------------------------------------------------------
# MathVision
# ---------------------------------------------------------------------------


class TestMathVisionTransform:
    def test_basic_transform(self):
        img = _mock_image()
        row = {
            "question": "Find the area",
            "decoded_image": img,
            "options": ["12", "24", "36", "48"],
            "answer": "24",
        }
        result = mathvision_transform(row)
        assert result["question"] == "Find the area"
        assert result["images"] == [img]
        assert result["choices"] == ["12", "24", "36", "48"]
        assert result["ground_truth"] == "24"
        assert result["data_source"] == "mathvision"

    def test_open_ended(self):
        img = _mock_image()
        row = {
            "question": "Calculate the value",
            "decoded_image": img,
            "options": [],
            "answer": "3.14",
        }
        result = mathvision_transform(row)
        assert result["choices"] == []

    def test_no_image(self):
        row = {
            "question": "Q",
            "options": [],
            "answer": "42",
        }
        result = mathvision_transform(row)
        assert result["images"] == []


# ---------------------------------------------------------------------------
# MathVista
# ---------------------------------------------------------------------------


class TestMathVistaTransform:
    def test_basic_transform(self):
        img = _mock_image()
        row = {
            "query": "What is the value of x in the figure?",
            "decoded_image": img,
            "answer": "5.0",
            "answer_type": "float",
            "choices": None,
        }
        result = mathvista_transform(row)
        assert result["question"] == "What is the value of x in the figure?"
        assert result["images"] == [img]
        assert result["ground_truth"] == "5.0"
        assert result["answer_type"] == "float"
        assert result["data_source"] == "mathvista"

    def test_choices_as_json_string(self):
        img = _mock_image()
        row = {
            "query": "Pick the answer",
            "decoded_image": img,
            "answer": "B",
            "answer_type": "text",
            "choices": '["one", "two", "three"]',
        }
        result = mathvista_transform(row)
        assert result["choices"] == ["one", "two", "three"]

    def test_integer_answer(self):
        img = _mock_image()
        row = {
            "query": "Count the items",
            "decoded_image": img,
            "answer": 7,
            "answer_type": "int",
            "choices": None,
        }
        result = mathvista_transform(row)
        assert result["ground_truth"] == "7"


# ---------------------------------------------------------------------------
# DynaMath
# ---------------------------------------------------------------------------


class TestDynaMathTransform:
    def test_basic_transform(self):
        img = _mock_image()
        row = {
            "question": "What is the slope?",
            "decoded_image": img,
            "ground_truth": "2.5",
            "answer_type": "float",
        }
        result = dynamath_transform(row)
        assert result["question"] == "What is the slope?"
        assert result["images"] == [img]
        assert result["ground_truth"] == "2.5"
        assert result["answer_type"] == "float"
        assert result["data_source"] == "dynamath"

    def test_image_field_fallback(self):
        img = _mock_image()
        row = {
            "question": "Q",
            "image": img,
            "ground_truth": "3",
            "answer_type": "int",
        }
        result = dynamath_transform(row)
        assert result["images"] == [img]

    def test_answer_fallback(self):
        img = _mock_image()
        row = {
            "question": "Q",
            "decoded_image": img,
            "answer": "42",
        }
        result = dynamath_transform(row)
        assert result["ground_truth"] == "42"


# ---------------------------------------------------------------------------
# ZEROBench
# ---------------------------------------------------------------------------


class TestZEROBenchTransform:
    def test_basic_transform(self):
        img1 = _mock_image()
        img2 = _mock_image()
        row = {
            "question_text": "What is happening in these images?",
            "question_images_decoded": [img1, img2],
            "question_answer": "A sunrise over mountains",
        }
        result = zerobench_transform(row)
        assert result["question"] == "What is happening in these images?"
        assert result["images"] == [img1, img2]
        assert result["ground_truth"] == "A sunrise over mountains"
        assert result["data_source"] == "zerobench"

    def test_single_image_not_list(self):
        img = _mock_image()
        row = {
            "question_text": "Q",
            "question_images_decoded": img,
            "question_answer": "A",
        }
        result = zerobench_transform(row)
        assert result["images"] == [img]

    def test_no_images(self):
        row = {
            "question_text": "Q",
            "question_answer": "A",
        }
        result = zerobench_transform(row)
        assert result["images"] == []


# ---------------------------------------------------------------------------
# ZEROBench Subquestions
# ---------------------------------------------------------------------------


class TestZEROBenchSubTransform:
    def test_basic_transform(self):
        img = _mock_image()
        row = {
            "question_text": "Count the objects",
            "question_images_decoded": [img],
            "question_answer": "7",
        }
        result = zerobench_sub_transform(row)
        assert result["question"] == "Count the objects"
        assert result["images"] == [img]
        assert result["ground_truth"] == "7"
        assert result["data_source"] == "zerobench_sub"


# ---------------------------------------------------------------------------
# VLMs Are Blind
# ---------------------------------------------------------------------------


class TestVLMsAreBlindTransform:
    def test_basic_transform(self):
        img = _mock_image()
        row = {
            "prompt": "How many circles are in the image?",
            "image": img,
            "groundtruth": "{5}",
        }
        result = vlmsareblind_transform(row)
        assert result["question"] == "How many circles are in the image?"
        assert result["images"] == [img]
        assert result["ground_truth"] == "5"
        assert result["data_source"] == "vlmsareblind"

    def test_complex_groundtruth(self):
        img = _mock_image()
        row = {
            "prompt": "Q",
            "image": img,
            "groundtruth": "{42}",
        }
        result = vlmsareblind_transform(row)
        assert result["ground_truth"] == "42"

    def test_groundtruth_without_braces(self):
        img = _mock_image()
        row = {
            "prompt": "Q",
            "image": img,
            "groundtruth": "plain answer",
        }
        result = vlmsareblind_transform(row)
        assert result["ground_truth"] == "plain answer"

    def test_no_image(self):
        row = {
            "prompt": "Q",
            "groundtruth": "{1}",
        }
        result = vlmsareblind_transform(row)
        assert result["images"] == []


# ---------------------------------------------------------------------------
# BabyVision
# ---------------------------------------------------------------------------


class TestBabyVisionTransform:
    def test_choice_answer(self):
        img = _mock_image()
        row = {
            "question": "Which animal is shown?",
            "image": img,
            "ansType": "choice",
            "choiceAns": 2,
            "blankAns": "",
            "choice0": "Cat",
            "choice1": "Dog",
            "choice2": "Bird",
            "choice3": "Fish",
        }
        result = babyvision_transform(row)
        assert result["question"] == "Which animal is shown?"
        assert result["images"] == [img]
        assert result["ground_truth"] == "C"  # index 2 → C
        assert result["choices"] == ["Cat", "Dog", "Bird", "Fish"]
        assert result["data_source"] == "babyvision"

    def test_blank_answer(self):
        img = _mock_image()
        row = {
            "question": "What color is the object?",
            "image": img,
            "ansType": "blank",
            "choiceAns": None,
            "blankAns": "red",
        }
        result = babyvision_transform(row)
        assert result["ground_truth"] == "red"
        assert result["choices"] is None

    def test_choice_index_zero(self):
        img = _mock_image()
        row = {
            "question": "Q",
            "image": img,
            "ansType": "choice",
            "choiceAns": 0,
            "choice0": "First",
            "choice1": "Second",
        }
        result = babyvision_transform(row)
        assert result["ground_truth"] == "A"

    def test_no_choices_fields(self):
        img = _mock_image()
        row = {
            "question": "Q",
            "image": img,
            "ansType": "blank",
            "blankAns": "answer",
        }
        result = babyvision_transform(row)
        assert result["choices"] is None


# ---------------------------------------------------------------------------
# Standard VLM output schema checks
# ---------------------------------------------------------------------------


class TestVLMOutputSchema:
    """Verify all VLM transforms produce the expected fields."""

    @pytest.mark.parametrize("transform_fn,row", [
        (mmmu_transform, {
            "question": "Q", "options": '["A", "B"]', "answer": "A", "subfield": "S",
        }),
        (mmmu_pro_transform, {
            "question": "Q", "options": '["A", "B"]', "answer": "A", "subject": "S",
        }),
        (mathvision_transform, {
            "question": "Q", "options": [], "answer": "42",
        }),
        (mathvista_transform, {
            "query": "Q", "answer": "42", "answer_type": "int", "choices": None,
        }),
        (dynamath_transform, {
            "question": "Q", "ground_truth": "42", "answer_type": "int",
        }),
        (zerobench_transform, {
            "question_text": "Q", "question_answer": "A",
        }),
        (zerobench_sub_transform, {
            "question_text": "Q", "question_answer": "A",
        }),
        (vlmsareblind_transform, {
            "prompt": "Q", "groundtruth": "{1}",
        }),
        (babyvision_transform, {
            "question": "Q", "ansType": "blank", "blankAns": "A",
        }),
    ])
    def test_has_required_fields(self, transform_fn, row):
        result = transform_fn(row)
        assert "question" in result
        assert "images" in result
        assert "ground_truth" in result
        assert "data_source" in result
        assert isinstance(result["images"], list)
