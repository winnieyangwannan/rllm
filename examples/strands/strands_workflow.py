import asyncio
from concurrent.futures import ThreadPoolExecutor
from rllm.agents.agent import Episode
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow
from rllm.integrations.strands import StrandsAgent


class StrandsWorkflow(Workflow):
    """Workflow for running StrandsAgent with proper trajectory tracking."""
    
    def __init__(
        self,
        agent_cls,
        agent_args=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Store agent creation parameters instead of creating the agent
        self.agent_cls = agent_cls
        self.agent_args = dict(agent_args) if agent_args is not None else {}
        
        # Initialize uid and task attributes
        self.uid = None
        self.task = None
        self.agent = None  # Will be created fresh for each task
    
    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        """Execute the Strands workflow."""
        
        # Set uid and task for the workflow
        self.uid = uid
        self.task = task
        
        # CRITICAL: Clear ALL workflow state before creating new agent
        # This ensures no contamination between tasks when workflow is reused
        if hasattr(self, '_agent_registry'):
            self._agent_registry.clear()
        if hasattr(self, '_completed_trajectories'):
            self._completed_trajectories.clear()
        # Clear any other potential state
        if hasattr(self, 'agents'):
            self.agents.clear()
        if hasattr(self, '_agents'):
            self._agents.clear()
        
        # Create fresh agent for each task (ensures zero contamination)
        task_text = task.get("task", "No task specified")
        self.agent = self.agent_cls(**self.agent_args)
        
        # Register the fresh agent
        self.register_agent(self.agent)
        
        # Set initial trajectory task (fresh agent, so no reset needed)
        self.agent.reset_trajectory(task=task_text)
        
        # Run the agent with the task
        result = self.agent(task_text)
        
        # Commit the agent's trajectory
        self.commit("strands_agent", self.agent, reset=False)
        
        # Check for termination
        if hasattr(result, "stop_reason") and result.stop_reason in ["end_turn", "stop_sequence"]:
            raise TerminationEvent(TerminationReason.ENV_DONE)
        
        # Default termination
        raise TerminationEvent(TerminationReason.ENV_DONE)
    
    def is_multithread_safe(self) -> bool:
        """Check if this workflow is safe for multithreading."""
        # StrandsAgent should be thread-safe for basic operations
        return True
