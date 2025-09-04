"""Gaia dataset evaluator using StrandsAgent and RLLM integration.

This module provides evaluation capabilities for the Gaia dataset using
the existing Strands + RLLM integration.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd

from rllm.integrations.strands import StrandsAgent, RLLMModel
from rllm.engine.rollout import OpenAIEngine


@dataclass
class GaiaSample:
    """Represents a single Gaia dataset sample."""
    task_id: str
    question: str
    answer: str
    file_name: str
    level: Optional[str] = None
    file_path: Optional[str] = None


@dataclass
class EvaluationResult:
    """Represents the evaluation result for a single sample."""
    task_id: str
    question: str
    ground_truth: str
    model_response: str
    is_correct: bool
    f1_score: float
    exact_match: bool
    tool_usage: Dict[str, int]
    execution_time: float
    error_message: Optional[str] = None


class GaiaEvaluator:
    """Evaluator for Gaia dataset using StrandsAgent."""
    
    def __init__(
        self,
        rollout_engine: OpenAIEngine,
        tools: List[Any],
        system_prompt: Optional[str] = None,
        max_samples: Optional[int] = None,
        output_dir: str = "outputs/gaia_eval"
    ):
        """Initialize the Gaia evaluator.
        
        Args:
            rollout_engine: The rollout engine for model inference
            tools: List of tools to provide to the agent
            system_prompt: System prompt for the agent
            max_samples: Maximum number of samples to evaluate (None for all)
            output_dir: Directory to save evaluation results
        """
        self.rollout_engine = rollout_engine
        self.tools = tools
        self.max_samples = max_samples
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model and agent
        self.model = RLLMModel(
            rollout_engine=rollout_engine, 
            model_id="gaia-evaluator"
        )
        
        default_system_prompt = (
            "You are a helpful agent solving web-based tasks. "
            "Use tools when beneficial to find information and solve problems. "
            "Provide clear, accurate answers based on the information you gather."
        )
        
        self.agent = StrandsAgent(
            model=self.model,
            tools=tools,
            system_prompt=system_prompt or default_system_prompt
        )
        
        self.results: List[EvaluationResult] = []
        self.tool_usage_stats: Dict[str, int] = {}
        
    def load_gaia_dataset(self, dataset_path: str) -> List[GaiaSample]:
        """Load Gaia dataset from JSON file.
        
        Args:
            dataset_path: Path to the Gaia dataset JSON file
            
        Returns:
            List of GaiaSample objects
        """
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            sample = GaiaSample(
                task_id=item.get('task_id', ''),
                question=item.get('problem', item.get('Question', '')),
                answer=item.get('tests', item.get('Final answer', '')),
                file_name=item.get('file_name', ''),
                level=item.get('Level', None),
                file_path=item.get('file_path', None)
            )
            samples.append(sample)
        
        if self.max_samples:
            samples = samples[:self.max_samples]
            
        print(f"Loaded {len(samples)} samples from {dataset_path}")
        return samples
    
    def evaluate_single_sample(self, sample: GaiaSample) -> EvaluationResult:
        """Evaluate a single Gaia sample.
        
        Args:
            sample: The GaiaSample to evaluate
            
        Returns:
            EvaluationResult with evaluation metrics
        """
        start_time = time.time()
        
        try:
            # Reset agent trajectory for new task
            self.agent.reset_trajectory(task=sample.question)
            
            # Run the agent
            result = self.agent(sample.question)
            
            # Extract model response
            model_response = self._extract_response(result)
            
            # Calculate metrics
            is_correct, f1_score, exact_match = self._calculate_metrics(
                model_response, sample.answer
            )
            
            # Get tool usage from trajectory
            tool_usage = self._extract_tool_usage()
            
            execution_time = time.time() - start_time
            
            return EvaluationResult(
                task_id=sample.task_id,
                question=sample.question,
                ground_truth=sample.answer,
                model_response=model_response,
                is_correct=is_correct,
                f1_score=f1_score,
                exact_match=exact_match,
                tool_usage=tool_usage,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return EvaluationResult(
                task_id=sample.task_id,
                question=sample.question,
                ground_truth=sample.answer,
                model_response="",
                is_correct=False,
                f1_score=0.0,
                exact_match=False,
                tool_usage={},
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def evaluate_dataset(self, dataset_path: str) -> List[EvaluationResult]:
        """Evaluate the entire Gaia dataset.
        
        Args:
            dataset_path: Path to the Gaia dataset JSON file
            
        Returns:
            List of EvaluationResult objects
        """
        samples = self.load_gaia_dataset(dataset_path)
        results = []
        
        print(f"Starting evaluation of {len(samples)} samples...")
        
        for i, sample in enumerate(samples):
            print(f"\n{'='*60}")
            print(f"📝 Sample {i+1}/{len(samples)}: {sample.task_id}")
            print(f"❓ Question: {sample.question}")
            print(f"✅ Ground Truth: {sample.answer}")
            
            result = self.evaluate_single_sample(sample)
            results.append(result)
            
            # Print model response and evaluation result
            print(f"🤖 Model Response: {result.model_response}")
            print(f"📊 Result: {'✅ Correct' if result.is_correct else '❌ Incorrect'}")
            print(f"🎯 F1 Score: {result.f1_score:.3f}")
            print(f"🎯 Exact Match: {'Yes' if result.exact_match else 'No'}")
            print(f"⏱️  Execution Time: {result.execution_time:.2f}s")
            if result.tool_usage:
                print(f"🔧 Tools Used: {result.tool_usage}")
            if result.error_message:
                print(f"⚠️  Error: {result.error_message}")
            print(f"{'='*60}")
            
            # Update tool usage stats
            for tool_name, count in result.tool_usage.items():
                self.tool_usage_stats[tool_name] = self.tool_usage_stats.get(tool_name, 0) + count
        
        self.results = results
        return results
    
    def _extract_response(self, result: Any) -> str:
        """Extract text response from agent result."""
        if hasattr(result, 'message'):
            if isinstance(result.message, dict):
                content = result.message.get('content', [])
            elif hasattr(result.message, 'content'):
                content = result.message.content
            else:
                content = []
            
            # Extract text content
            if isinstance(content, list):
                text_parts = []
                for event in content:
                    if isinstance(event, dict) and 'text' in event:
                        text_parts.append(event['text'])
                return ' '.join(text_parts)
            else:
                return str(content)
        
        return str(result)
    
    def _calculate_metrics(self, model_response: str, ground_truth: str) -> Tuple[bool, float, bool]:
        """Calculate evaluation metrics."""
        # Simple exact match
        exact_match = model_response.strip().lower() == ground_truth.strip().lower()
        
        # Simple F1 score calculation (can be enhanced)
        if exact_match:
            f1_score = 1.0
        else:
            # Basic F1 calculation - can be improved with more sophisticated text similarity
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
        
        # Consider correct if F1 > 0.5 or exact match
        is_correct = exact_match or f1_score > 0.5
        
        return is_correct, f1_score, exact_match
    
    def _extract_tool_usage(self) -> Dict[str, int]:
        """Extract tool usage statistics from agent trajectory."""
        tool_usage = {}
        
        if hasattr(self.agent, 'trajectory') and self.agent.trajectory:
            for step in self.agent.trajectory.steps:
                if hasattr(step, 'action') and step.action:
                    # Extract tool name from action if available
                    if hasattr(step.action, 'tool_name'):
                        tool_name = step.action.tool_name
                        tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        
        return tool_usage
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        if not self.results:
            return {"error": "No results to report"}
        
        total_samples = len(self.results)
        correct_samples = sum(1 for r in self.results if r.is_correct)
        exact_matches = sum(1 for r in self.results if r.exact_match)
        
        # Calculate average metrics
        avg_f1 = sum(r.f1_score for r in self.results) / total_samples
        avg_execution_time = sum(r.execution_time for r in self.results) / total_samples
        
        # Error analysis
        errors = [r for r in self.results if r.error_message]
        
        report = {
            "summary": {
                "total_samples": total_samples,
                "correct_samples": correct_samples,
                "accuracy": correct_samples / total_samples,
                "exact_match_rate": exact_matches / total_samples,
                "average_f1_score": avg_f1,
                "average_execution_time": avg_execution_time
            },
            "tool_usage": self.tool_usage_stats,
            "error_count": len(errors),
            "results": [asdict(r) for r in self.results]
        }
        
        return report
    
    def save_results(self, filename: Optional[str] = None):
        """Save evaluation results to files."""
        if not filename:
            timestamp = int(time.time())
            filename = f"gaia_eval_results_{timestamp}"
        
        # Save detailed results as JSON
        json_path = os.path.join(self.output_dir, f"{filename}.json")
        with open(json_path, 'w') as f:
            json.dump(self.generate_report(), f, indent=2)
        
        # Save summary as CSV
        csv_path = os.path.join(self.output_dir, f"{filename}_summary.csv")
        summary_data = []
        for result in self.results:
            summary_data.append({
                'task_id': result.task_id,
                'is_correct': result.is_correct,
                'f1_score': result.f1_score,
                'exact_match': result.exact_match,
                'execution_time': result.execution_time,
                'error_message': result.error_message or ''
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
        
        return json_path, csv_path
