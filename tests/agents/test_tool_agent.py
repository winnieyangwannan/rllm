import json
from unittest.mock import Mock, patch

import pytest

from rllm.agents.agent import Step, Trajectory
from rllm.agents.tool_agent import ToolAgent
from rllm.tools.tool_base import Tool


class MockTool(Tool):
    """Mock tool for testing."""

    def __init__(self, name="mock_tool"):
        self.name = name
        self.description = "A mock tool for testing"
        self.parameters = {"type": "object", "properties": {"query": {"type": "string", "description": "Test query"}}, "required": ["query"]}

    @property
    def json(self):
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def call(self, **kwargs):
        return f"Mock tool called with {kwargs}"


class TestToolAgent:
    """Test suite for ToolAgent class."""

    def test_init_default(self):
        """Test ToolAgent initialization with default parameters."""
        agent = ToolAgent()
        assert agent.system_prompt is not None
        assert agent.tools is not None
        assert agent.tool_parser is not None
        assert isinstance(agent._trajectory, Trajectory)
        assert len(agent.messages) == 1  # System message
        assert agent.step == 0
        assert agent.messages[0]["role"] == "system"

    @patch("rllm.agents.tool_agent.get_tool_parser")
    @patch("rllm.tools.multi_tool.tool_registry")
    def test_init_with_tools_list(self, mock_registry, mock_get_parser):
        """Test ToolAgent initialization with tools list."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        # Mock the registry to contain our test tools
        mock_registry.__contains__ = Mock(return_value=True)
        mock_registry.instantiate = Mock(return_value=MockTool())

        agent = ToolAgent(tools=["calculator", "search"])
        assert agent.tools is not None
        mock_get_parser.assert_called_once_with(parser_name="qwen")

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_init_with_tool_map(self, mock_get_parser):
        """Test ToolAgent initialization with tool_map."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        tool_map = {"mock_tool": MockTool}
        agent = ToolAgent(tool_map=tool_map)
        assert agent.tools is not None

    def test_init_with_both_tools_and_tool_map_raises_error(self):
        """Test that providing both tools and tool_map raises ValueError."""
        with pytest.raises(ValueError, match="Cannot specify both 'tools' and 'tool_map' parameters"):
            ToolAgent(tools=["calculator"], tool_map={"mock": MockTool})

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_reset(self, mock_get_parser):
        """Test the reset method."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()
        initial_message_count = len(agent.messages)

        # Add some state to reset
        agent.messages.append({"role": "user", "content": "test"})
        agent.step = 5
        agent._trajectory.steps.append(Step())

        agent.reset()

        assert isinstance(agent._trajectory, Trajectory)
        assert agent._trajectory.steps == []
        assert len(agent.messages) == initial_message_count  # Should have system message
        assert agent.step == 0
        assert agent.messages[0]["role"] == "system"

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_chat_completions_property(self, mock_get_parser):
        """Test the chat_completions property."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()
        assert agent.chat_completions == agent.messages
        assert isinstance(agent.chat_completions, list)

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_trajectory_property(self, mock_get_parser):
        """Test the trajectory property."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()
        assert agent.trajectory == agent._trajectory
        assert isinstance(agent.trajectory, Trajectory)

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_format_observation_as_messages_dict_with_question(self, mock_get_parser):
        """Test _format_observation_as_messages with dict containing question."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()
        obs = {"question": "What is the weather?"}
        messages = agent._format_observation_as_messages(obs)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is the weather?"

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_format_observation_as_messages_dict_with_tool_outputs(self, mock_get_parser):
        """Test _format_observation_as_messages with dict containing tool_outputs."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()
        obs = {"tool_outputs": {"call_1": "Weather is sunny", "call_2": "Temperature is 25°C"}}
        messages = agent._format_observation_as_messages(obs)

        assert len(messages) == 2
        for msg in messages:
            assert msg["role"] == "tool"
            assert "tool_call_id" in msg
            assert msg["content"] in ["Weather is sunny", "Temperature is 25°C"]

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_format_observation_as_messages_string(self, mock_get_parser):
        """Test _format_observation_as_messages with string observation."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()
        obs = "Hello, how can I help?"
        messages = agent._format_observation_as_messages(obs)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, how can I help?"

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_update_from_env_initial_observation(self, mock_get_parser):
        """Test update_from_env with initial observation."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()
        initial_message_count = len(agent.messages)

        observation = {"question": "What's the weather today?"}
        reward = 0.0
        done = False
        info = {}

        agent.update_from_env(observation, reward, done, info)

        # Check that message was added
        assert len(agent.messages) == initial_message_count + 1
        assert agent.messages[-1]["role"] == "user"
        assert agent.messages[-1]["content"] == "What's the weather today?"

        # Check that step was added to trajectory
        assert len(agent.trajectory.steps) == 1
        current_step = agent.trajectory.steps[0]
        assert current_step.observation == observation
        assert current_step.step == 0

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_update_from_env_with_prior_step(self, mock_get_parser):
        """Test update_from_env when there's a prior step."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()

        # Add initial step
        initial_step = Step(observation="initial", step=0)
        agent._trajectory.steps.append(initial_step)

        observation = {"tool_outputs": {"call_1": "Tool result"}}
        reward = 0.5
        done = False
        info = {"step": 2}

        agent.update_from_env(observation, reward, done, info)

        # Check that prior step was updated
        prior_step = agent._trajectory.steps[0]
        assert prior_step.next_observation == observation
        assert prior_step.reward == reward
        assert prior_step.done == done
        assert prior_step.info == info

        # Check that tool output message was added
        assert any(msg["role"] == "tool" for msg in agent.messages)

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_update_from_env_done_episode(self, mock_get_parser):
        """Test update_from_env when episode is done."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()

        # Add initial step
        initial_step = Step(observation="initial", step=0)
        agent._trajectory.steps.append(initial_step)

        observation = "Task completed"
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

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_update_from_model_with_valid_tool_calls(self, mock_get_parser):
        """Test update_from_model with valid tool calls."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"

        # Mock tool call parsing
        mock_tool_call = Mock()
        mock_tool_call.to_dict.return_value = {"name": "search", "arguments": {"query": "weather"}}
        mock_parser_instance.parse.return_value = [mock_tool_call]

        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()

        # Add initial step
        step = Step(observation="test", step=0)
        agent._trajectory.steps.append(step)

        response = "I'll search for the weather information."
        agent.update_from_model(response)

        # Check that step was updated
        current_step = agent.get_current_state()
        assert current_step.model_response == response
        assert isinstance(current_step.action, list)
        assert len(current_step.action) == 1
        assert current_step.action[0]["type"] == "function"

        # Check that message was added
        assert agent.messages[-1]["role"] == "assistant"
        assert agent.messages[-1]["content"] == response

        # Check step counter
        assert agent.step == 1

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_update_from_model_with_parsing_error(self, mock_get_parser):
        """Test update_from_model when tool parsing fails."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"

        # Mock parsing error
        mock_parser_instance.parse.side_effect = Exception("Parsing failed")

        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()

        # Add initial step
        step = Step(observation="test", step=0)
        agent._trajectory.steps.append(step)

        response = "This is a response without tool calls."
        agent.update_from_model(response)

        # Should create a finish tool call
        current_step = agent.get_current_state()
        assert isinstance(current_step.action, list)
        assert len(current_step.action) == 1
        assert current_step.action[0]["function"]["name"] == "finish"

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_update_from_model_empty_trajectory_raises_error(self, mock_get_parser):
        """Test that update_from_model raises error when trajectory is empty."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()

        with pytest.raises(ValueError, match="update_from_model called before update_from_env after reset"):
            agent.update_from_model("test response")

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_get_current_state(self, mock_get_parser):
        """Test get_current_state method."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()

        # Add a step
        step = Step(observation="test", step=0)
        agent._trajectory.steps.append(step)

        current_state = agent.get_current_state()
        assert current_state == step

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_full_interaction_flow(self, mock_get_parser):
        """Test a complete interaction flow."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"

        # Mock tool call
        mock_tool_call = Mock()
        mock_tool_call.to_dict.return_value = {"name": "search", "arguments": {"query": "weather"}}
        mock_parser_instance.parse.return_value = [mock_tool_call]

        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()
        initial_message_count = len(agent.messages)

        # Step 1: Initial environment update
        observation1 = {"question": "What's the weather?"}
        agent.update_from_env(observation1, 0.0, False, {})

        assert len(agent.trajectory.steps) == 1
        assert len(agent.messages) == initial_message_count + 1

        # Step 2: Model response with tool call
        response1 = "I'll search for weather information."
        agent.update_from_model(response1)

        assert agent.step == 1
        assert len(agent.messages) == initial_message_count + 2
        current_step = agent.get_current_state()
        assert isinstance(current_step.action, list)

        # Step 3: Environment feedback with tool results
        tool_outputs = {"tool_outputs": {"call_1": "Weather is sunny, 25°C"}}
        agent.update_from_env(tool_outputs, 0.0, False, {})

        # Should have tool message
        assert any(msg["role"] == "tool" for msg in agent.messages)

        # Step 4: Final model response
        response2 = "The weather is sunny and 25°C."
        agent.update_from_model(response2)

        # Step 5: Final feedback
        agent.update_from_env("task_complete", 1.0, True, {"success": True})

        assert len(agent.trajectory.steps) == 2
        assert agent.step == 2

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_tool_calls_with_dict_arguments(self, mock_get_parser):
        """Test tool calls with dictionary arguments get converted to JSON strings."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"

        # Mock tool call with dict arguments
        mock_tool_call = Mock()
        mock_tool_call.to_dict.return_value = {"name": "complex_search", "arguments": {"filters": {"date": "2023", "type": "news"}}}
        mock_parser_instance.parse.return_value = [mock_tool_call]

        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()

        # Add initial step
        step = Step(observation="test", step=0)
        agent._trajectory.steps.append(step)

        response = "I'll perform a complex search."
        agent.update_from_model(response)

        # Check that dict arguments were converted to JSON string
        current_step = agent.get_current_state()
        tool_call = current_step.action[0]
        arguments = tool_call["function"]["arguments"]
        assert isinstance(arguments, str)
        # Should be valid JSON
        parsed_args = json.loads(arguments)
        assert "filters" in parsed_args

    @patch("rllm.agents.tool_agent.get_tool_parser")
    def test_trajectory_to_dict(self, mock_get_parser):
        """Test that trajectory can be converted to dict."""
        mock_parser_class = Mock()
        mock_parser_instance = Mock()
        mock_parser_instance.get_tool_prompt.return_value = "tool prompt"
        mock_parser_instance.parse.return_value = []
        mock_parser_class.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_class

        agent = ToolAgent()

        # Add some interaction
        agent.update_from_env({"question": "test"}, 0.0, False, {})
        agent.update_from_model("test response")
        agent.update_from_env("feedback", 1.0, True, {})

        trajectory_dict = agent.trajectory.to_dict()
        assert isinstance(trajectory_dict, dict)
        assert "steps" in trajectory_dict
        assert "reward" in trajectory_dict
        assert isinstance(trajectory_dict["steps"], list)
