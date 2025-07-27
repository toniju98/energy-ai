import pytest
import numpy as np
from env import ThermostatEnv

def test_environment_initialization():
    """Test that the environment initializes with correct default values"""
    env = ThermostatEnv()
    assert env.temperature is not None
    assert env.day == 0
    assert env.hour == 0
    assert env.energy_used == 0
    assert isinstance(env.temperature, float)
    assert isinstance(env.day, int)
    assert isinstance(env.hour, int)

def test_step_function():
    """Test that the step function returns correct observation space"""
    env = ThermostatEnv()
    state, reward = env.step(0)  # Assuming 0 is a valid action
    assert isinstance(state, tuple)
    assert len(state) == 4  # temperature, day, hour, external_temp
    assert isinstance(reward, float)

def test_reset_function():
    """Test that the environment resets properly"""
    env = ThermostatEnv()
    # Since there's no explicit reset function, we'll test the initial state
    assert env.temperature == 22
    assert env.day == 0
    assert env.hour == 0
    assert env.energy_used == 0

def test_temperature_bounds():
    """Test that temperature stays within reasonable bounds"""
    env = ThermostatEnv()
    # Run for several steps with extreme actions
    for _ in range(100):
        state, _ = env.step(1)  # Heating action
        assert state[0] <= 30  # Max temperature
        state, _ = env.step(2)  # Cooling action
        assert state[0] >= 15  # Min temperature

def test_energy_consumption():
    """Test that energy consumption is always non-negative"""
    env = ThermostatEnv()
    for _ in range(10):
        _, _ = env.step(1)  # Heating action
        assert env.energy_used >= 0 