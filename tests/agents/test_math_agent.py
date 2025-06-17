import pytest
from unittest.mock import Mock, patch
from rllm.agents.math_agent import MathAgent
from rllm.agents.agent import Step, Trajectory


class TestMathAgent:
    """Test suite for MathAgent class."""

    def test_init_default(self):
        """Test MathAgent initialization with default parameters."""
        agent = MathAgent()
        assert agent.accumulate_thinking is True
        assert agent.instruction == "Let's think step by step. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
        assert isinstance(agent._trajectory, Trajectory)
        assert agent.messages == []
        assert agent.step == 0

    def test_init_custom_accumulate_thinking(self):
        """Test MathAgent initialization with custom accumulate_thinking parameter."""
        agent = MathAgent(accumulate_thinking=False)
        assert agent.accumulate_thinking is False

    def test_reset(self):
        """Test the reset method."""
        agent = MathAgent()
        # Add some state to reset
        agent.messages = [{"role": "user", "content": "test"}]
        agent.step = 5
        agent._trajectory.steps.append(Step())
        
        agent.reset()
        
        assert isinstance(agent._trajectory, Trajectory)
        assert agent._trajectory.steps == []
        assert agent.messages == []
        assert agent.step == 0

    def test_chat_completions_property(self):
        """Test the chat_completions property."""
        agent = MathAgent()
        test_messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}
        ]
        agent.messages = test_messages
        
        assert agent.chat_completions == test_messages

    def test_trajectory_property(self):
        """Test the trajectory property."""
        agent = MathAgent()
        assert agent.trajectory == agent._trajectory
        assert isinstance(agent.trajectory, Trajectory)

    def test_update_from_env_initial_question(self):
        """Test update_from_env with initial question observation."""
        agent = MathAgent()
        observation = {"question": "What is the square root of 16?"}
        reward = 0.0
        done = False
        info = {}
        
        agent.update_from_env(observation, reward, done, info)
        
        # Check that message was added correctly
        assert len(agent.messages) == 1
        expected_content = f"What is the square root of 16? {agent.instruction}"
        assert agent.messages[0]["role"] == "user"
        assert agent.messages[0]["content"] == expected_content
        
        # Check that step was added to trajectory
        assert len(agent.trajectory.steps) == 1
        current_step = agent.trajectory.steps[0]
        assert current_step.observation == expected_content
        assert current_step.step == 0

    def test_update_from_env_follow_up_correction(self):
        """Test update_from_env with follow-up correction."""
        agent = MathAgent()
        
        # First, add an initial step
        initial_step = Step(observation="initial", step=0)
        agent._trajectory.steps.append(initial_step)
        
        observation = "correction needed"
        reward = 0.5
        done = False
        info = {"attempt": 2}
        
        agent.update_from_env(observation, reward, done, info)
        
        # Check that prior step was updated
        prior_step = agent._trajectory.steps[0]
        expected_correction = ("Your previous answer may contain a mistake. "
                             "Please review it carefully and answer again. "
                             "Put your final answer within \\boxed{}.")
        assert prior_step.next_observation == expected_correction
        assert prior_step.reward == reward
        assert prior_step.done == done
        assert prior_step.info == info
        
        # Check that new message and step were added
        assert len(agent.messages) == 1
        assert agent.messages[0]["content"] == expected_correction
        assert len(agent.trajectory.steps) == 2

    def test_update_from_env_done_episode(self):
        """Test update_from_env when episode is done."""
        agent = MathAgent()
        
        # Add initial step
        initial_step = Step(observation="initial", step=0)
        agent._trajectory.steps.append(initial_step)
        
        observation = "final"
        reward = 1.0
        done = True
        info = {"final": True}
        
        agent.update_from_env(observation, reward, done, info)
        
        # Should update prior step but not add new step since done=True
        prior_step = agent._trajectory.steps[0]
        assert prior_step.reward == reward
        assert prior_step.done == done
        assert prior_step.info == info
        
        # Should not add new step when done=True
        assert len(agent.trajectory.steps) == 1

    def test_update_from_model_with_accumulate_thinking(self):
        """Test update_from_model with accumulate_thinking=True."""
        agent = MathAgent(accumulate_thinking=True)
        
        # Add initial step
        step = Step(observation="test", step=0)
        agent._trajectory.steps.append(step)
        
        response = "<think>Let me calculate this</think> The answer is 4"
        agent.update_from_model(response)
        
        # Check that step was updated
        current_step = agent.get_current_state()
        assert current_step.model_response == response
        assert current_step.action == response
        
        # Check that message was added
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "assistant"
        assert agent.messages[0]["content"] == response
        
        # Check step counter
        assert agent.step == 1

    def test_update_from_model_without_accumulate_thinking(self):
        """Test update_from_model with accumulate_thinking=False."""
        agent = MathAgent(accumulate_thinking=False)
        
        # Add initial step
        step = Step(observation="test", step=0)
        agent._trajectory.steps.append(step)
        
        response = "<think>Let me calculate this</think> The answer is 4"
        agent.update_from_model(response)
        
        # Check that thinking portion was removed from message
        expected_content = " The answer is 4"
        assert agent.messages[0]["content"] == expected_content
        
        # But original response should be stored in model_response
        current_step = agent.get_current_state()
        assert current_step.model_response == response

    def test_update_from_model_no_thinking_tags(self):
        """Test update_from_model when response has no thinking tags."""
        agent = MathAgent(accumulate_thinking=False)
        
        # Add initial step
        step = Step(observation="test", step=0)
        agent._trajectory.steps.append(step)
        
        response = "The answer is 4"
        agent.update_from_model(response)
        
        # Should keep original response since no thinking tags
        assert agent.messages[0]["content"] == response

    def test_update_from_model_empty_trajectory_raises_error(self):
        """Test that update_from_model raises error when trajectory is empty."""
        agent = MathAgent()
        
        with pytest.raises(AssertionError):
            agent.update_from_model("test response")

    def test_get_current_state(self):
        """Test get_current_state method."""
        agent = MathAgent()
        
        # Add a step
        step = Step(observation="test", step=0)
        agent._trajectory.steps.append(step)
        
        current_state = agent.get_current_state()
        assert current_state == step

    def test_get_current_state_empty_trajectory_raises_error(self):
        """Test that get_current_state raises error when trajectory is empty."""
        agent = MathAgent()
        
        with pytest.raises(AssertionError):
            agent.get_current_state()

    def test_full_interaction_flow(self):
        """Test a complete interaction flow."""
        agent = MathAgent()
        
        # Step 1: Initial environment update
        observation1 = {"question": "What is 5 + 3?"}
        agent.update_from_env(observation1, 0.0, False, {})
        
        assert len(agent.trajectory.steps) == 1
        assert len(agent.messages) == 1
        
        # Step 2: Model response
        response1 = "<think>5 + 3 = 8</think> \\boxed{8}"
        agent.update_from_model(response1)
        
        assert agent.step == 1
        assert len(agent.messages) == 2
        current_step = agent.get_current_state()
        assert current_step.action == response1
        
        # Step 3: Environment feedback
        agent.update_from_env("correct", 1.0, True, {"correct": True})
        
        # Prior step should be updated with reward
        assert current_step.reward == 1.0
        assert current_step.done is True

    def test_multiple_rounds_interaction(self):
        """Test multiple rounds of interaction."""
        agent = MathAgent()
        
        # Round 1
        agent.update_from_env({"question": "What is 2 + 2?"}, 0.0, False, {})
        agent.update_from_model("The answer is 5")  # Wrong answer
        
        # Round 2 - correction
        agent.update_from_env("incorrect", 0.0, False, {})
        agent.update_from_model("The answer is \\boxed{4}")  # Correct answer
        
        # Final feedback
        agent.update_from_env("correct", 1.0, True, {})
        
        assert len(agent.trajectory.steps) == 2
        assert agent.step == 2
        
        # Check that correction prompt was used
        correction_message = agent.messages[2]["content"]
        assert "Your previous answer may contain a mistake" in correction_message

    def test_trajectory_to_dict(self):
        """Test that trajectory can be converted to dict."""
        agent = MathAgent()
        
        # Add some interaction
        agent.update_from_env({"question": "test"}, 0.0, False, {})
        agent.update_from_model("test response")
        agent.update_from_env("feedback", 1.0, True, {})
        
        trajectory_dict = agent.trajectory.to_dict()
        assert isinstance(trajectory_dict, dict)
        assert "steps" in trajectory_dict
        assert "reward" in trajectory_dict
        assert isinstance(trajectory_dict["steps"], list)
