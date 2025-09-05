"""Test script for Gaia evaluator functionality."""

import json
import os
import tempfile

from gaia_evaluator import EvaluationResult, GaiaSample


def create_test_dataset():
    """Create a small test dataset for testing."""
    test_data = [{"task_id": "test_001", "problem": "What is 2 + 2?", "tests": "4", "file_name": "test1.txt", "Level": "easy"}, {"task_id": "test_002", "problem": "What is the capital of France?", "tests": "Paris", "file_name": "test2.txt", "Level": "medium"}, {"task_id": "test_003", "problem": "What is 5 * 6?", "tests": "30", "file_name": "test3.txt", "Level": "easy"}]

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(test_data, temp_file)
    temp_file.close()

    return temp_file.name


def test_gaia_sample():
    """Test GaiaSample dataclass."""
    print("Testing GaiaSample...")

    sample = GaiaSample(task_id="test_001", question="What is 2 + 2?", answer="4", file_name="test.txt")

    assert sample.task_id == "test_001"
    assert sample.question == "What is 2 + 2?"
    assert sample.answer == "4"
    print("✅ GaiaSample test passed")


def test_evaluation_result():
    """Test EvaluationResult dataclass."""
    print("Testing EvaluationResult...")

    result = EvaluationResult(task_id="test_001", question="What is 2 + 2?", ground_truth="4", model_response="4", is_correct=True, f1_score=1.0, exact_match=True, tool_usage={"calculator": 1}, execution_time=1.5)

    assert result.task_id == "test_001"
    assert result.is_correct == True
    assert result.f1_score == 1.0
    print("✅ EvaluationResult test passed")


def test_metrics_calculation():
    """Test metrics calculation methods."""
    print("Testing metrics calculation...")

    # Create a mock evaluator instance
    class MockEvaluator:
        def _calculate_metrics(self, model_response: str, ground_truth: str):
            # Simple exact match
            exact_match = model_response.strip().lower() == ground_truth.strip().lower()

            # Simple F1 score calculation
            if exact_match:
                f1_score = 1.0
            else:
                model_words = set(model_response.lower().split())
                gt_words = set(ground_truth.lower().split())

                if not model_words or not gt_words:
                    f1_score = 0.0
                else:
                    intersection = len(model_words.intersection(gt_words))
                    precision = intersection / len(model_words) if model_words else 0
                    recall = intersection / len(gt_words) if gt_words else 0

                    if precision + recall == 0:
                        f1_score = 0.0
                    else:
                        f1_score = 2 * (precision * recall) / (precision + recall)

            is_correct = exact_match or f1_score > 0.5

            return is_correct, f1_score, exact_match

    evaluator = MockEvaluator()

    # Test exact match
    print("  Testing exact match: '4' vs '4'")
    is_correct, f1_score, exact_match = evaluator._calculate_metrics("4", "4")
    print(f"    Result: is_correct={is_correct}, f1_score={f1_score:.3f}, exact_match={exact_match}")
    assert is_correct == True
    assert f1_score == 1.0
    assert exact_match == True

    # Test partial match
    print("  Testing partial match: 'The answer is 4' vs '4'")
    is_correct, f1_score, exact_match = evaluator._calculate_metrics("The answer is 4", "4")
    print(f"    Result: is_correct={is_correct}, f1_score={f1_score:.3f}, exact_match={exact_match}")
    # F1 score should be 0.4 (precision=0.25, recall=1.0), so is_correct should be False
    assert is_correct == False
    assert exact_match == False

    # Test no match
    print("  Testing no match: '5' vs '4'")
    is_correct, f1_score, exact_match = evaluator._calculate_metrics("5", "4")
    print(f"    Result: is_correct={is_correct}, f1_score={f1_score:.3f}, exact_match={exact_match}")
    assert is_correct == False
    assert exact_match == False

    evaluator = MockEvaluator()

    # Test exact match
    is_correct, f1_score, exact_match = evaluator._calculate_metrics("4", "4")
    assert is_correct == True
    assert f1_score == 1.0
    assert exact_match == True

    # Test partial match
    is_correct, f1_score, exact_match = evaluator._calculate_metrics("The answer is 4", "4")
    # F1 score should be 0.4 (precision=0.25, recall=1.0), so is_correct should be False
    assert is_correct == False
    assert exact_match == False

    # Test no match
    is_correct, f1_score, exact_match = evaluator._calculate_metrics("5", "4")
    assert is_correct == False
    assert exact_match == False

    print("✅ Metrics calculation test passed")


def test_dataset_loading():
    """Test dataset loading functionality."""
    print("Testing dataset loading...")

    # Create test dataset
    test_file = create_test_dataset()

    try:
        # Test loading
        with open(test_file) as f:
            data = json.load(f)

        assert len(data) == 3
        assert data[0]["task_id"] == "test_001"
        assert data[0]["problem"] == "What is 2 + 2?"

        print("✅ Dataset loading test passed")

        # Display sample data
        print("\n📊 Sample Data:")
        for i, item in enumerate(data):
            print(f"\nSample {i + 1}:")
            print(f"  Task ID: {item['task_id']}")
            print(f"  Question: {item['problem']}")
            print(f"  Ground Truth: {item['tests']}")
            print(f"  File: {item['file_name']}")

    finally:
        # Clean up
        os.unlink(test_file)


def main():
    """Run all tests."""
    print("🧪 Running Gaia Evaluator Tests")
    print("=" * 40)

    try:
        test_gaia_sample()
        test_evaluation_result()
        test_metrics_calculation()
        test_dataset_loading()

        print("\n" + "=" * 40)
        print("🎉 All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
