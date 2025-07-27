import pytest
import torch
import numpy as np
from dqn_agent import DQNAgent

def test_agent_initialization():
    """Test that the DQN agent initializes with correct components"""
    agent = DQNAgent()
    assert hasattr(agent, 'memory')
    assert hasattr(agent, 'exploration_rate')
    assert isinstance(agent.memory, (list, tuple, type(agent.memory)))
    assert isinstance(agent.exploration_rate, float)
    assert agent.exploration_rate > 0
    assert agent.exploration_rate <= 1

def test_choose_action_function():
    """Test that the choose_action function returns valid actions"""
    agent = DQNAgent()
    state = np.random.random(agent.input_size)
    action = agent.choose_action(state)
    assert isinstance(action, int)
    assert 0 <= action < agent.output_size

def test_store_experience_function():
    """Test that the memory buffer works correctly"""
    agent = DQNAgent()
    state = np.random.random(agent.input_size)
    action = 0
    reward = 1.0
    next_state = np.random.random(agent.input_size)
    initial_memory_size = len(agent.memory)
    agent.store_experience(state, action, reward, next_state)
    assert len(agent.memory) == initial_memory_size + 1
    memory_item = agent.memory[-1]
    assert len(memory_item) == 4
    assert np.allclose(memory_item[0], state)
    assert memory_item[1] == action
    assert memory_item[2] == reward
    assert np.allclose(memory_item[3], next_state)

def test_update_networks_function():
    """Test that the update_networks function does not crash with enough memory"""
    agent = DQNAgent()
    # Add enough experiences to memory
    for _ in range(agent.batch_size):
        state = np.random.random(agent.input_size)
        action = np.random.randint(0, agent.output_size)
        reward = np.random.random()
        next_state = np.random.random(agent.input_size)
        agent.store_experience(state, action, reward, next_state)
    # Get initial weights
    initial_weights = {k: v.clone() for k, v in agent.policy_net.state_dict().items()}
    agent.update_networks()
    # Get new weights
    new_weights = agent.policy_net.state_dict()
    # Check that weights have changed
    weights_changed = any(not torch.equal(initial_weights[k], new_weights[k]) for k in initial_weights)
    assert weights_changed, "Network weights should change after update_networks()"

def test_exploration_rate_decay():
    """Test that exploration_rate decays over time"""
    agent = DQNAgent()
    initial_exploration_rate = agent.exploration_rate
    agent.exploration_rate *= agent.exploration_decay
    assert agent.exploration_rate < initial_exploration_rate
    assert agent.exploration_rate >= 0 