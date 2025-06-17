import pytest
from unittest.mock import Mock, patch
from rllm.agents.frozenlake_agent import FrozenLakeAgent
from rllm.agents.agent import Step, Trajectory


class TestFrozenLakeAgent:
    """Test suite for FrozenLakeAgent class."""

    def test_init_default(self):
        """Test FrozenLakeAgent initialization with default parameters."""
        agent = FrozenLakeAgent()
        assert isinstance(agent._trajectory, Trajectory)
        assert len(agent.messages) == 1  # System message
        assert agent.step == 0
        assert agent.accumulate_thinking is False
        assert agent.multistep_prompt is False
        assert agent.max_steps is None
        assert agent.messages[0]["role"] == "system"
        assert "FrozenLake Quick Guide" in agent.messages[0]["content"]

    def test_init_with_max_steps(self):
        """Test FrozenLakeAgent initialization with max_steps parameter."""
        agent = FrozenLakeAgent(max_steps=20)
        assert agent.max_steps == 20

    def test_reset(self):
        """Test the reset method."""
        agent = FrozenLakeAgent()
        initial_message_content = agent.messages[0]["content"]
        
        # Add some state to reset
        agent.messages.append({"role": "user", "content": "test"})
        agent.step = 5
        agent._trajectory.steps.append(Step())
        
        agent.reset()
        
        assert isinstance(agent._trajectory, Trajectory)
        assert agent._trajectory.steps == []
        assert len(agent.messages) == 1  # Should have system message
        assert agent.step == 0
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[0]["content"] == initial_message_content

    def test_chat_completions_property(self):
        """Test the chat_completions property."""
        agent = FrozenLakeAgent()
        assert agent.chat_completions == agent.messages
        assert isinstance(agent.chat_completions, list)

    def test_trajectory_property(self):
        """Test the trajectory property."""
        agent = FrozenLakeAgent()
        assert agent.trajectory == agent._trajectory
        assert isinstance(agent.trajectory, Trajectory)

    def test_multistep_prompt_setting(self):
        """Test the multistep_prompt setting affects system prompt."""
        agent = FrozenLakeAgent()
        agent.multistep_prompt = True
        agent.reset()
        
        assert "Below are examples for an interaction:" in agent.messages[0]["content"]

    def test_update_from_env_initial_observation(self):
        """Test update_from_env with initial observation."""
        agent = FrozenLakeAgent()
        initial_message_count = len(agent.messages)
        
        observation = "P _ _ G\n_ O _ _\n_ _ _ _\n_ _ _ _"
        reward = 0.0
        done = False
        info = {}
        
        agent.update_from_env(observation, reward, done, info)
        
        # Check that message was added correctly
        assert len(agent.messages) == initial_message_count + 1
        assert agent.messages[-1]["role"] == "user"
        expected_content = f"Current Observation (0): \n{observation}\nYou have not achieved the goal, P has not reached G yet. Please give the next action."
        assert agent.messages[-1]["content"] == expected_content
        
        # Check that step was added to trajectory
        assert len(agent.trajectory.steps) == 1
        current_step = agent.trajectory.steps[0]
        assert current_step.observation == str(observation)
        assert current_step.step == 0

    def test_update_from_env_with_max_steps(self):
        """Test update_from_env includes max steps information."""
        agent = FrozenLakeAgent(max_steps=10)
        
        observation = "P _ _ G"
        agent.update_from_env(observation, 0.0, False, {})
        
        message_content = agent.messages[-1]["content"]
        assert "The maximum number of steps remaining is 10." in message_content

    def test_update_from_env_same_observation_warning(self):
        """Test update_from_env adds warning when observation doesn't change."""
        agent = FrozenLakeAgent()
        
        # First observation
        observation = "P _ _ G"
        agent.update_from_env(observation, 0.0, False, {})
        agent.update_from_model("I'll move right. ```Right```")
        
        # Same observation (indicating invalid move)
        agent.update_from_env(observation, 0.0, False, {})
        
        message_content = agent.messages[-1]["content"]
        assert "Your last response is invalid" in message_content
        assert "Your position didn't change at all" in message_content

    def test_update_from_env_with_prior_step(self):
        """Test update_from_env when there's a prior step."""
        agent = FrozenLakeAgent()
        
        # Add initial step with action
        initial_step = Step(observation="initial", step=0, action="1")  # action=1 means it's a completed step
        agent._trajectory.steps.append(initial_step)
        
        observation = "P _ _ G"
        reward = 0.5
        done = False
        info = {"step": 2}
        
        agent.update_from_env(observation, reward, done, info)
        
        # Check that prior step was updated
        prior_step = agent._trajectory.steps[0]
        assert prior_step.next_observation == str(observation)
        assert prior_step.reward == reward
        assert prior_step.done == done
        assert prior_step.info == info

    def test_update_from_env_done_episode(self):
        """Test update_from_env when episode is done."""
        agent = FrozenLakeAgent()
        
        # Add initial step
        initial_step = Step(observation="initial", step=0)
        agent._trajectory.steps.append(initial_step)
        
        observation = "G"  # Reached goal
        reward = 1.0
        done = True
        info = {"success": True}
        
        agent.update_from_env(observation, reward, done, info)
        
        # Should update prior step but not add new step since done=True
        prior_step = agent._trajectory.steps[0]
        assert prior_step.reward == reward
        assert prior_step.done == done
        assert prior_step.info == info
        
        # Should not add new step when done=True
        assert len(agent.trajectory.steps) == 1

    def test_parse_model_response_valid_direction(self):
        """Test _parse_model_response with valid direction."""
        agent = FrozenLakeAgent()
        
        response = "I need to move right to reach the goal. ```Right```"
        thought, action = agent._parse_model_response(response)
        
        assert thought == "I need to move right to reach the goal."
        assert action == "3"  # Right = 3

    def test_parse_model_response_valid_direction_case_insensitive(self):
        """Test _parse_model_response with different cases."""
        agent = FrozenLakeAgent()
        
        test_cases = [
            ("```up```", "4"),
            ("```DOWN```", "2"),
            ("```Left```", "1"),
            ("```RIGHT```", "3")
        ]
        
        for response_action, expected_action in test_cases:
            response = f"Moving. {response_action}"
            thought, action = agent._parse_model_response(response)
            assert action == expected_action

    def test_parse_model_response_numeric_action(self):
        """Test _parse_model_response with numeric action."""
        agent = FrozenLakeAgent()
        
        response = "I'll move up. ```4```"
        thought, action = agent._parse_model_response(response)
        
        assert action == "4"

    def test_parse_model_response_invalid_action(self):
        """Test _parse_model_response with invalid action."""
        agent = FrozenLakeAgent()
        
        response = "I'll move diagonally. ```diagonal```"
        thought, action = agent._parse_model_response(response)
        
        # Should return INVALID_ACTION (which is 0)
        assert action == "0"  # FrozenLakeEnv.INVALID_ACTION is 0

    def test_parse_model_response_no_code_blocks(self):
        """Test _parse_model_response with no code blocks."""
        agent = FrozenLakeAgent()
        
        response = "I need to think about this move."
        thought, action = agent._parse_model_response(response)
        
        assert thought == response
        # Should return INVALID_ACTION when no action is found (which is 0)
        assert action == "0"  # FrozenLakeEnv.INVALID_ACTION is 0

    def test_parse_model_response_multiple_code_blocks(self):
        """Test _parse_model_response uses the last code block."""
        agent = FrozenLakeAgent()
        
        response = "First I thought ```left``` but actually ```right``` is better."
        thought, action = agent._parse_model_response(response)
        
        assert "First I thought" in thought
        assert thought == "First I thought ```left``` but actually"  # Should stop before the last code block
        assert action == "3"  # Right = 3

    def test_update_from_model_with_accumulate_thinking_false(self):
        """Test update_from_model with accumulate_thinking=False."""
        agent = FrozenLakeAgent()
        agent.accumulate_thinking = False
        
        # Add initial step
        step = Step(observation="test", step=0)
        agent._trajectory.steps.append(step)
        
        response = "<think>Let me analyze this</think>I'll move right. ```Right```"
        agent.update_from_model(response)
        
        # Check that thinking portion was removed from message
        expected_content = "I'll move right. ```Right```"
        assert agent.messages[-1]["content"] == expected_content
        
        # Original response should also be stored in model_response for FrozenLakeAgent (different from MathAgent)
        current_step = agent.get_current_state()
        assert current_step.model_response == expected_content

    def test_update_from_model_stores_parsed_action(self):
        """Test update_from_model correctly stores parsed action and thought."""
        agent = FrozenLakeAgent()
        
        # Add initial step
        step = Step(observation="test", step=0)
        agent._trajectory.steps.append(step)
        
        response = "I need to move up to avoid the hole. ```Up```"
        agent.update_from_model(response)
        
        # Check that step was updated
        current_step = agent.get_current_state()
        assert current_step.model_response == response
        assert current_step.action == "4"  # Up = 4
        assert current_step.thought == "I need to move up to avoid the hole."
        
        # Check that message was added
        assert agent.messages[-1]["role"] == "assistant"
        assert agent.messages[-1]["content"] == response
        
        # Check step counter
        assert agent.step == 1

    def test_update_from_model_empty_trajectory_raises_error(self):
        """Test that update_from_model raises error when trajectory is empty."""
        agent = FrozenLakeAgent()
        
        with pytest.raises(AssertionError):
            agent.update_from_model("test response")

    def test_get_current_state(self):
        """Test get_current_state method."""
        agent = FrozenLakeAgent()
        
        # Add a step
        step = Step(observation="test", step=0)
        agent._trajectory.steps.append(step)
        
        current_state = agent.get_current_state()
        assert current_state == step

    def test_get_current_state_empty_trajectory_raises_error(self):
        """Test that get_current_state raises error when trajectory is empty."""
        agent = FrozenLakeAgent()
        
        with pytest.raises(ValueError, match="get_current_state called before the first observation was processed"):
            agent.get_current_state()

    def test_compute_training_reward(self):
        """Test compute_training_reward method."""
        agent = FrozenLakeAgent()
        
        # Test with empty trajectory
        empty_trajectory = Trajectory()
        assert agent.compute_training_reward(empty_trajectory) == 0
        
        # Test with trajectory containing steps
        trajectory = Trajectory()
        step1 = Step(reward=0.5)
        step2 = Step(reward=1.0)
        trajectory.steps = [step1, step2]
        
        assert agent.compute_training_reward(trajectory) == 1.0  # Last step's reward

    def test_validate_step(self):
        """Test validate_step method."""
        agent = FrozenLakeAgent()
        
        # Test valid action
        valid_step = Step(action="1")  # Left
        assert agent.validate_step(valid_step) is True
        
        # Test invalid action (INVALID_ACTION is 0)
        invalid_step = Step(action="0")  # FrozenLakeEnv.INVALID_ACTION
        assert agent.validate_step(invalid_step) is False

    def test_full_interaction_flow(self):
        """Test a complete interaction flow."""
        agent = FrozenLakeAgent()
        
        # Step 1: Initial environment update
        observation1 = "P _ _ G\n_ O _ _\n_ _ _ _\n_ _ _ _"
        agent.update_from_env(observation1, 0.0, False, {})
        
        assert len(agent.trajectory.steps) == 1
        assert len(agent.messages) == 2  # System + user message
        
        # Step 2: Model response
        response1 = "I need to move right to get closer to the goal. ```Right```"
        agent.update_from_model(response1)
        
        assert agent.step == 1
        assert len(agent.messages) == 3  # System + user + assistant
        current_step = agent.get_current_state()
        assert current_step.action == "3"  # Right
        assert current_step.thought == "I need to move right to get closer to the goal."
        
        # Step 3: Environment feedback
        observation2 = "_ P _ G\n_ O _ _\n_ _ _ _\n_ _ _ _"  # Agent moved right
        agent.update_from_env(observation2, 0.0, False, {})
        
        # Prior step should be updated with new observation
        assert current_step.next_observation == observation2
        assert current_step.reward == 0.0
        assert current_step.done is False

    def test_multiple_rounds_with_invalid_move(self):
        """Test multiple rounds including an invalid move."""
        agent = FrozenLakeAgent()
        
        # Round 1: Valid move
        observation1 = "P _ G"
        agent.update_from_env(observation1, 0.0, False, {})
        agent.update_from_model("Moving right. ```Right```")
        
        # Round 2: Invalid move (same observation)
        agent.update_from_env(observation1, 0.0, False, {})  # Same observation
        
        # Check that warning message was added
        warning_message = agent.messages[-1]["content"]
        assert "Your last response is invalid" in warning_message
        
        # Agent tries again
        agent.update_from_model("Let me try up instead. ```Up```")
        
        # Round 3: Successful move
        observation2 = "_ _ G\nP _ _"  # Agent moved up
        agent.update_from_env(observation2, 0.0, False, {})
        
        assert len(agent.trajectory.steps) == 3  # Each env update creates a new step
        assert agent.step == 2

    def test_goal_reached_scenario(self):
        """Test scenario where agent reaches the goal."""
        agent = FrozenLakeAgent()
        
        # Start near goal
        observation1 = "_ P G"
        agent.update_from_env(observation1, 0.0, False, {})
        agent.update_from_model("I can reach the goal by moving right. ```Right```")
        
        # Reach goal
        observation2 = "_ _ PG"  # P and G overlap
        agent.update_from_env(observation2, 1.0, True, {"success": True})
        
        # Check final state
        current_step = agent.get_current_state()
        assert current_step.reward == 1.0
        assert current_step.done is True
        assert current_step.info["success"] is True

    def test_trajectory_to_dict(self):
        """Test that trajectory can be converted to dict."""
        agent = FrozenLakeAgent()
        
        # Add some interaction
        agent.update_from_env("P _ G", 0.0, False, {})
        agent.update_from_model("Moving right. ```Right```")
        agent.update_from_env("_ P G", 1.0, True, {"success": True})
        
        trajectory_dict = agent.trajectory.to_dict()
        assert isinstance(trajectory_dict, dict)
        assert "steps" in trajectory_dict
        assert "reward" in trajectory_dict
        assert isinstance(trajectory_dict["steps"], list)

    def test_max_steps_countdown(self):
        """Test that max_steps countdown works correctly."""
        agent = FrozenLakeAgent(max_steps=3)
        
        # Step 1
        agent.update_from_env("P _ G", 0.0, False, {})
        assert "steps remaining is 3" in agent.messages[-1]["content"]
        agent.update_from_model("```Right```")
        
        # Step 2
        agent.update_from_env("_ P G", 0.0, False, {})
        assert "steps remaining is 2" in agent.messages[-1]["content"]
        agent.update_from_model("```Right```")
        
        # Step 3
        agent.update_from_env("_ _ PG", 1.0, True, {"success": True})
        # No remaining steps message when done=True
        
        assert agent.step == 2  # 0-indexed

    def test_process_action_for_validation(self):
        """Test _process_action_for_validation method."""
        agent = FrozenLakeAgent()
        
        response = "I'll move up. ```Up```"
        action = agent._process_action_for_validation(response)
        assert action == "4"  # Up = 4
