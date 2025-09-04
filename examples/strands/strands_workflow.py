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
        
        # Initialize mutable defaults
        agent_args = dict(agent_args) if agent_args is not None else {}
        
        # Create and register the agent
        self.agent = agent_cls(**agent_args)
        self.register_agent(self.agent)
        
        # Initialize uid and task attributes
        self.uid = None
        self.task = None
    
    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        """Execute the Strands workflow."""
        
        # Set uid and task for the workflow
        self.uid = uid
        self.task = task
        
        # Reset agent trajectory for new task
        task_text = task.get("task", "No task specified")
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
