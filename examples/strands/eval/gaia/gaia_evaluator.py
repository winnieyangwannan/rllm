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
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from strands.handlers.callback_handler import null_callback_handler
import sys
import os
from concurrent.futures import ThreadPoolExecutor
# Add parent directory to path to import strands_workflow
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from strands_workflow import StrandsWorkflow


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
        
        # Initialize model
        self.model = RLLMModel(
            rollout_engine=rollout_engine, 
            model_id="gaia-evaluator"
        )
        
        default_system_prompt = (
            "You are a helpful agent solving web-based tasks. "
            "Use tools when beneficial to find information and solve problems. "
            "Provide clear, accurate answers based on the information you gather."
        )
        
        # Following terminus_workflow.py pattern: store base parameters for workflow creation
        # We create fresh model instances per task to ensure zero contamination
        self.base_workflow_args = {
            "agent_cls": StrandsAgent,
            "base_agent_args": {
                "tools": tools,
                "system_prompt": system_prompt or default_system_prompt,
                "callback_handler": null_callback_handler
            }
        }
        
        self.results: List[EvaluationResult] = []
        self.tool_usage_stats: Dict[str, int] = {}
        
        # Store parameters for reference
        self._rollout_engine = rollout_engine
        self._tools = tools
        self._system_prompt = system_prompt or default_system_prompt
        
    def _create_fresh_agent(self) -> 'StrandsAgent':
        """Create a completely fresh agent instance for maximum isolation.
        
        This is the most thorough way to ensure no state pollution between tasks.
        Use this if you experience any cross-task contamination issues.
        """
        # Create new model instance
        fresh_model = RLLMModel(
            rollout_engine=self._rollout_engine, 
            model_id="gaia-evaluator-fresh"
        )
        
        # Create new agent instance
        fresh_agent = StrandsAgent(
            model=fresh_model,
            tools=self._tools,
            system_prompt=self._system_prompt,
            callback_handler=null_callback_handler
        )
        
        return fresh_agent
        
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
    
    async def evaluate_single_sample(self, sample: GaiaSample) -> EvaluationResult:
        """Evaluate a single Gaia sample using AgentWorkflowEngine.
        
        Args:
            sample: The GaiaSample to evaluate
            
        Returns:
            EvaluationResult with evaluation metrics
        """
        start_time = time.time()
        
        try:
            # Create fresh workflow for each task (GAIA-specific approach for zero contamination)
            # This ensures complete isolation between evaluation samples
            task_dict = {"task": sample.question}
            task_id = f"gaia_task_{sample.task_id}"
            
            # Following terminus_workflow.py pattern: create fresh workflow with fresh model
            # This ensures complete zero contamination for GAIA evaluation
            fresh_model = RLLMModel(
                rollout_engine=self._rollout_engine, 
                model_id=f"gaia-evaluator-{sample.task_id}"
            )
            
            workflow_args = {
                "agent_cls": self.base_workflow_args["agent_cls"],
                "agent_args": {
                    "model": fresh_model,
                    **self.base_workflow_args["base_agent_args"]
                }
            }
            
            fresh_workflow = StrandsWorkflow(
                rollout_engine=self._rollout_engine,
                executor=ThreadPoolExecutor(max_workers=1),
                **workflow_args
            )
            
            # Execute task with fresh workflow
            episode = await fresh_workflow.run_with_termination_handling(task_dict, task_id)
            
            if not episode:
                raise Exception("No episode returned from workflow")
            
            # Extract trajectory data from episode (same as run_strands.py save_episode_to_json)
            trajectory = episode.trajectories[0][1] if episode.trajectories else None
            if not trajectory:
                raise Exception("No trajectory found in episode")
            
            # Extract model response from trajectory steps
            model_response = self._extract_response_from_trajectory(trajectory)
            
            # Calculate metrics
            is_correct, f1_score, exact_match = self._calculate_metrics(
                model_response, sample.answer
            )
            
            # Get tool usage from trajectory
            tool_usage = self._extract_tool_usage_from_trajectory(trajectory)
            
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
    
    async def evaluate_dataset(self, dataset_path: str) -> List[EvaluationResult]:
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
            print()  # Clean separator
            
            result = await self.evaluate_single_sample(sample)
            results.append(result)
            
            # Print model response and evaluation result
            print(f"🤖 Model Response: {result.model_response}")
            print(f"📊 Result: {'✅ Correct' if result.is_correct else '❌ Incorrect'} | F1: {result.f1_score:.3f} | Exact Match: {'Yes' if result.exact_match else 'No'}")
            print(f"⏱️  Time: {result.execution_time:.2f}s", end="")
            if result.tool_usage:
                tools_summary = ", ".join([f"{tool}({count})" for tool, count in result.tool_usage.items()])
                print(f" | 🔧 Tools: {tools_summary}")
            else:
                print()
            if result.error_message:
                print(f"⚠️  Error: {result.error_message}")
            print(f"{'='*60}")
            
            # Update tool usage stats
            for tool_name, count in result.tool_usage.items():
                self.tool_usage_stats[tool_name] = self.tool_usage_stats.get(tool_name, 0) + count
        
        self.results = results
        return results
    
    def _extract_response_from_trajectory(self, trajectory) -> str:
        """Extract final text response from trajectory steps."""
        if not trajectory.steps:
            return ""
        
        # Look through all steps to find the final model response
        final_response = ""
        
        for step in trajectory.steps:
            # Check step.model_response first
            if step.model_response and step.model_response.strip():
                if not (step.model_response.startswith('[Tool:') or 
                       step.model_response.startswith('[Using tool:')):
                    final_response = step.model_response
            
            # Check step.action['final_response'] (where actual response is stored)
            if (step.action and isinstance(step.action, dict) and 
                'final_response' in step.action):
                final_resp = step.action['final_response']
                
                if isinstance(final_resp, dict) and 'content' in final_resp:
                    # Extract text from content array
                    content = final_resp['content']
                    if isinstance(content, list):
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and 'text' in item:
                                text = item['text']
                                # Skip tool call representations
                                if not (text.startswith('[Tool:') or text.startswith('[Using tool:')):
                                    text_parts.append(text)
                        if text_parts:
                            final_response = ' '.join(text_parts)
                elif isinstance(final_resp, str):
                    final_response = final_resp
        
        return final_response
    
    def _extract_tool_usage_from_trajectory(self, trajectory) -> Dict[str, int]:
        """Extract tool usage statistics from trajectory steps."""
        tool_usage = {}
        
        for step in trajectory.steps:
            if hasattr(step, 'action') and step.action and isinstance(step.action, dict):
                # Handle new tool_calls format (hybrid architecture)
                if step.action.get("type") == "tool_calls":
                    for tool_call in step.action.get("tool_calls", []):
                        tool_name = tool_call.get("name")
                        if tool_name:
                            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
                # Handle legacy tool_call format (backward compatibility)
                elif step.action.get("type") == "tool_call":
                    tool_name = step.action.get("tool_name")
                    if tool_name:
                        tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        
        return tool_usage
    
    def _extract_response(self, result: Any) -> str:
        """Extract text response from agent result."""
        # Try to get the final text response from agent's chat_completions
        if hasattr(self.agent, 'chat_completions'):
            chat_completions = self.agent.chat_completions
            if chat_completions:
                # Get the last assistant message
                for msg in reversed(chat_completions):
                    if msg.get('role') == 'assistant' and msg.get('content', '').strip():
                        content = msg.get('content', '')
                        # Skip messages that only contain tool usage representations
                        if not content.startswith('[Tool:') and not content.startswith('[Using tool:'):
                            return content
        
        # Fallback to original extraction method
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
                if hasattr(step, 'action') and step.action and isinstance(step.action, dict):
                    # Handle new tool_calls format (hybrid architecture)
                    if step.action.get("type") == "tool_calls":
                        for tool_call in step.action.get("tool_calls", []):
                            tool_name = tool_call.get("name")
                            if tool_name:
                                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
                    # Handle legacy tool_call format (backward compatibility)
                    elif step.action.get("type") == "tool_call":
                        tool_name = step.action.get("tool_name")
                        if tool_name:
                            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
                    # Handle object-based action (fallback)
                    elif hasattr(step.action, 'tool_name'):
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
