"""Engine module for rLLM.

This module contains the core execution infrastructure for agent trajectory rollout.
"""

from .agent_execution_engine import AgentExecutionEngine, AsyncAgentExecutionEngine
from .agent_workflow_engine import AgentWorkflowEngine
from .rollout.openai_engine import OpenAIEngine
from .rollout.rollout_engine import RolloutEngine

# Local guard: only import VerlEngine if available to avoid requiring verl.experimental during inference
try:
    from .rollout.verl_engine import VerlEngine  # type: ignore
    _HAS_VERL = True
except Exception:  # pragma: no cover
    VerlEngine = None  # type: ignore
    _HAS_VERL = False

__all__ = [
    "AgentExecutionEngine",
    "AsyncAgentExecutionEngine",
    "AgentWorkflowEngine",
    "RolloutEngine",
    "OpenAIEngine",
]

if _HAS_VERL:
    __all__.append("VerlEngine")
