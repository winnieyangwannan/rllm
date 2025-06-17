import os
from unittest.mock import MagicMock, patch

import pytest

# Assuming rllm and its dependencies are importable
from rllm.environments.swe.swe import R2E_ENV_IDS, BatchSWEEnv, SWEEnv

# Use a smaller, faster dataset for testing
TEST_DATASET = "R2E-Gym/SWE-Bench-Lite"
TEST_SPLIT = "test"
TEST_BATCH_SIZE = 2
# Limit the dataset size to speed up tests and reduce resource usage
TRUNCATE_SIZE = 2

# Ensure the test dataset is valid
if TEST_DATASET not in R2E_ENV_IDS:
    pytest.skip(f"Test dataset {TEST_DATASET} not available", allow_module_level=True)

# Ensure docker group is enabled.
os.system("newgrp docker")


@pytest.fixture(scope="module")
def batch_swe_env():
    """Fixture to create and teardown a BatchSWEEnv instance for testing."""
    print(f"\nSetting up BatchSWEEnv with dataset: {TEST_DATASET}, split: {TEST_SPLIT}, batch_size: {TEST_BATCH_SIZE}, truncate: {TRUNCATE_SIZE}")
    # Mock load_dataset if you want fully isolated tests without downloading
    # with patch('rllm.environments.swe.swe.load_dataset') as mock_load_dataset:
    #     mock_load_dataset.return_value = ... # return a mock Dataset object

    # Use specific seeds for reproducibility if needed
    seeds = [i for i in range(TEST_BATCH_SIZE)]
    env = BatchSWEEnv(batch_size=TEST_BATCH_SIZE, dataset_name=TEST_DATASET, split=TEST_SPLIT, seeds=seeds, truncate_dataset_size=TRUNCATE_SIZE)
    yield env
    print("\nTearing down BatchSWEEnv...")
    # Mock os.system during close to prevent actual docker commands if needed
    with patch("os.system") as mock_system:
        env.close()
        # Assert that docker cleanup commands were attempted
        mock_system.assert_any_call("docker stop $(docker ps -a -q)")
        mock_system.assert_any_call("docker rm $(docker ps -a -q)")
    print("Teardown complete.")


def test_initialization(batch_swe_env: BatchSWEEnv):
    """Test BatchSWEEnv initialization."""
    assert batch_swe_env.batch_size == TEST_BATCH_SIZE
    assert len(batch_swe_env.envs) == TEST_BATCH_SIZE
    assert batch_swe_env.dataset_name == TEST_DATASET
    assert batch_swe_env.split == TEST_SPLIT
    assert batch_swe_env.seeds == list(range(TEST_BATCH_SIZE))
    for env in batch_swe_env.envs:
        assert isinstance(env, SWEEnv)
        # Check if dataset was truncated
        assert len(env.dataset) == TRUNCATE_SIZE


def test_reset(batch_swe_env: BatchSWEEnv):
    """Test resetting the batched environments."""
    observations, infos = batch_swe_env.reset(seed=42)

    assert isinstance(observations, list)
    assert len(observations) == TEST_BATCH_SIZE
    for obs in observations:
        # Initial observation should be the task instruction string
        assert isinstance(obs, str)
        assert len(obs) > 0  # Should not be empty

    assert isinstance(infos, list)
    assert len(infos) == TEST_BATCH_SIZE
    for info in infos:
        # Currently returns empty dicts
        assert isinstance(info, dict)


def test_step_all_envs(batch_swe_env: BatchSWEEnv):
    """Test stepping all environments in the batch."""
    batch_swe_env.reset(seed=123)  # Ensure envs are reset
    action_1 = """
    <function=file_editor>
      <parameter=command>view</parameter>
      <parameter=path>./sympy/tensor/array/dense_ndim_array.py</parameter>
      <parameter=concise>True</parameter>
    </function>
    """

    # Example actions - using search which is usually safe
    actions = [action_1 for i in range(TEST_BATCH_SIZE)]

    observations, rewards, terminateds, truncateds, infos = batch_swe_env.step(actions)

    assert isinstance(observations, list)
    assert len(observations) == TEST_BATCH_SIZE
    for obs in observations:
        assert isinstance(obs, str)  # Search results or empty string

    assert isinstance(rewards, list)
    assert len(rewards) == TEST_BATCH_SIZE
    for reward in rewards:
        assert isinstance(reward, float | int)  # Should be 0 unless terminated

    assert isinstance(terminateds, list)
    assert len(terminateds) == TEST_BATCH_SIZE
    for term in terminateds:
        assert isinstance(term, bool)

    assert isinstance(truncateds, list)
    assert len(truncateds) == TEST_BATCH_SIZE
    for trunc in truncateds:
        assert isinstance(trunc, bool)
        assert not trunc  # Truncation logic is within SWEEnv, not BatchedEnv directly

    assert isinstance(infos, list)
    assert len(infos) == TEST_BATCH_SIZE
    for info in infos:
        assert isinstance(info, dict)


def test_step_subset_envs(batch_swe_env: BatchSWEEnv):
    """Test stepping a subset of environments."""
    if TEST_BATCH_SIZE < 2:
        pytest.skip("Test requires batch size >= 2")

    batch_swe_env.reset(seed=456)  # Ensure envs are reset

    # Step only the first environment
    env_idxs_to_step = [0]
    action_2 = """
    <function=execute_bash>
      <parameter=command>search_dir</parameter>
      <parameter=search_term>class ImmutableDenseNDimArray</parameter>
    </function>
    """
    actions = [action_2]

    observations, rewards, terminateds, truncateds, infos = batch_swe_env.step(actions, env_idxs=env_idxs_to_step)

    # Check that results are returned only for the stepped environment
    assert len(observations) == len(env_idxs_to_step)
    assert len(rewards) == len(env_idxs_to_step)
    assert len(terminateds) == len(env_idxs_to_step)
    assert len(truncateds) == len(env_idxs_to_step)
    assert len(infos) == len(env_idxs_to_step)

    assert isinstance(observations[0], str)
    assert isinstance(rewards[0], float | int)
    assert isinstance(terminateds[0], bool)
    assert isinstance(truncateds[0], bool)
    assert isinstance(infos[0], dict)


def test_step_invalid_action_format(batch_swe_env: BatchSWEEnv):
    """Test stepping with an action string that doesn't parse."""
    batch_swe_env.reset(seed=111)
    env_idxs_to_step = [0]
    actions = ["this is not a valid action format"]  # Should be handled by Action.from_string

    # Expecting it might return empty observation, 0 reward, not done.
    observations, rewards, terminateds, truncateds, infos = batch_swe_env.step(actions, env_idxs=env_idxs_to_step)

    assert observations == [""]  # SWEEnv returns "" for invalid action name
    assert rewards == [0.0]
    assert terminateds == [False]
    assert truncateds == [False]
    assert infos == [{}]


def test_step_mismatched_actions_and_indices(batch_swe_env: BatchSWEEnv):
    """Test error handling for mismatched actions and indices."""
    batch_swe_env.reset(seed=222)

    # Provide 2 actions but only 1 index
    with pytest.raises(AssertionError) as excinfo:
        batch_swe_env.step(['search("a")', 'search("b")'], env_idxs=[0])
    assert "Number of actions" in str(excinfo.value)
    assert "must match the env used" in str(excinfo.value)

    # Provide 1 action but step all (implicitly requires TEST_BATCH_SIZE actions)
    if TEST_BATCH_SIZE > 1:
        with pytest.raises(AssertionError) as excinfo:
            batch_swe_env.step(['search("a")'])  # Missing env_idxs means step all
        assert "Number of actions must match batch size" in str(excinfo.value)


@patch("rllm.environments.swe.swe.BatchSWEEnv.__init__", return_value=None)  # Mock init
@patch("rllm.environments.swe.swe.SWEEnv")  # Mock the inner env too
def test_from_json(mock_swe_env, mock_batch_init):
    """Test creating BatchSWEEnv from JSON."""
    extra_infos = [{"seed": 10, "some_other_key": "value1"}, {"seed": 20, "another_key": "value2"}]

    # reconstructed_env = BatchSWEEnv.from_json(extra_infos)

    # Check that __init__ was called with arguments derived from extra_infos
    mock_batch_init.assert_called_once()
    args, kwargs = mock_batch_init.call_args

    assert kwargs.get("batch_size") == len(extra_infos)
    assert kwargs.get("seeds") == [10, 20]
    # Check default dataset/split were likely passed (or add them to extra_infos if needed)
    assert "dataset_name" in kwargs
    assert "split" in kwargs


# Test closing separately to potentially mock os.system without affecting other tests
@patch("os.system")
def test_close_calls_docker_cleanup(mock_os_system):
    """Test that close attempts to stop and remove docker containers."""
    # Need to create a separate instance as the fixture manages close
    seeds = [i for i in range(TEST_BATCH_SIZE)]
    env = BatchSWEEnv(batch_size=TEST_BATCH_SIZE, dataset_name=TEST_DATASET, split=TEST_SPLIT, seeds=seeds, truncate_dataset_size=TRUNCATE_SIZE)
    # Mock the individual env close methods to speed up
    for i in range(TEST_BATCH_SIZE):
        env.envs[i].close = MagicMock()

    env.close()

    # Verify individual envs were closed
    for i in range(TEST_BATCH_SIZE):
        env.envs[i].close.assert_called_once()

    # Verify docker commands were called
    mock_os_system.assert_any_call("docker stop $(docker ps -a -q)")
    mock_os_system.assert_any_call("docker rm $(docker ps -a -q)")
