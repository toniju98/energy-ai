import numpy as np
import matplotlib.pyplot as plt
from agent import QLearningAgent
from dqn_agent import DQNAgent
import argparse

def plot_training_results(rewards, exploration_rates, energy_usage, window_size=100):
    """
    Plot comprehensive training results including rewards, exploration rate, and energy usage.
    
    Args:
        rewards (list): List of rewards from training
        exploration_rates (list): List of exploration rates during training
        energy_usage (list): List of energy usage per episode
        window_size (int): Size of the moving average window
    """
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Rewards over time
    plt.subplot(3, 2, 1)
    plt.plot(rewards, label='Reward per Episode', alpha=0.3, color='blue')
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(moving_avg, label=f'Moving Average ({window_size} episodes)', 
             linewidth=2, color='red')
    plt.title('Training Progress - Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Exploration rate over time
    plt.subplot(3, 2, 2)
    plt.plot(exploration_rates, color='green')
    plt.title('Exploration Rate Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Exploration Rate (ε)')
    plt.grid(True)
    
    # Plot 3: Energy usage over time
    plt.subplot(3, 2, 3)
    plt.plot(energy_usage, color='orange', alpha=0.5, label='Energy per Episode')
    energy_moving_avg = np.convolve(energy_usage, np.ones(window_size)/window_size, mode='valid')
    plt.plot(energy_moving_avg, color='red', linewidth=2, 
             label=f'Moving Average ({window_size} episodes)')
    plt.title('Energy Usage Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Energy Units Used')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Reward distribution
    plt.subplot(3, 2, 4)
    plt.hist(rewards, bins=50, color='purple', alpha=0.7)
    plt.title('Reward Distribution')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Plot 5: Energy usage distribution
    plt.subplot(3, 2, 5)
    plt.hist(energy_usage, bins=50, color='orange', alpha=0.7)
    plt.title('Energy Usage Distribution')
    plt.xlabel('Energy Units Used')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Plot 6: Moving average of rewards
    plt.subplot(3, 2, 6)
    plt.plot(moving_avg, color='orange', linewidth=2)
    plt.title('Smoothed Reward Trend')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_test_results(test_episodes_data):
    """
    Plot the results of test episodes.
    
    Args:
        test_episodes_data (list): List of dictionaries containing test episode data
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
    
    # Plot temperature changes
    for i, episode in enumerate(test_episodes_data):
        temperatures = [step['temperature'] for step in episode['steps']]
        external_temps = [step['external_temp'] for step in episode['steps']]
        hours = range(len(temperatures))
        ax1.plot(hours, temperatures, label=f'Indoor Temp Episode {i+1}', marker='o')
        ax1.plot(hours, external_temps, '--', label=f'External Temp Episode {i+1}', alpha=0.5)
    
    ax1.axhspan(20, 22, color='green', alpha=0.2, label='Comfort Zone')
    ax1.set_title('Temperature Changes During Test Episodes')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Temperature (°C)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot actions taken
    action_colors = ['gray', 'red', 'blue']
    action_labels = ['Nothing', 'Heating', 'Cooling']
    
    for i, episode in enumerate(test_episodes_data):
        actions = [step['action'] for step in episode['steps']]
        hours = range(len(actions))
        ax2.scatter(hours, [i+1]*len(hours), c=[action_colors[a] for a in actions], 
                   marker='s', s=100)
    
    ax2.set_title('Actions Taken During Test Episodes')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Episode')
    ax2.set_yticks(range(1, len(test_episodes_data) + 1))
    ax2.set_yticklabels([f'Episode {i+1}' for i in range(len(test_episodes_data))])
    ax2.grid(True)
    
    # Add legend for actions
    for i, (color, label) in enumerate(zip(action_colors, action_labels)):
        ax2.scatter([], [], c=color, label=label, marker='s')
    ax2.legend()
    
    # Plot cumulative energy usage
    for i, episode in enumerate(test_episodes_data):
        # Calculate hourly energy usage
        energy_usage = [1 if step['action'] in [1, 2] else 0 for step in episode['steps']]
        cumulative_energy = np.cumsum(energy_usage)
        hours = range(len(cumulative_energy))
        
        # Plot cumulative energy
        ax3.plot(hours, cumulative_energy, label=f'Episode {i+1}', marker='o')
        
        # Add hourly energy usage as bars
        ax3.bar(hours, energy_usage, alpha=0.2, color='orange')
    
    ax3.set_title('Energy Usage During Test Episodes\n(Bars: Hourly Usage, Lines: Cumulative Usage)')
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Energy Units')
    ax3.grid(True)
    ax3.legend()
    
    # Add text showing total energy usage for each episode
    for i, episode in enumerate(test_episodes_data):
        total_energy = sum(1 for step in episode['steps'] if step['action'] in [1, 2])
        ax3.text(len(episode['steps'])-1, total_energy, 
                f'Total: {total_energy}', 
                ha='right', va='bottom')
    
    # Plot temperature difference (indoor - external)
    for i, episode in enumerate(test_episodes_data):
        temp_diffs = [step['temperature'] - step['external_temp'] 
                     for step in episode['steps']]
        hours = range(len(temp_diffs))
        ax4.plot(hours, temp_diffs, label=f'Episode {i+1}', marker='o')
    
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_title('Temperature Difference (Indoor - External)')
    ax4.set_xlabel('Hour')
    ax4.set_ylabel('Temperature Difference (°C)')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

def train_agent(agent, episodes):
    """Train the agent and return training metrics"""
    total_rewards = []
    exploration_rates = []
    energy_usage_per_episode = []
    
    print("Starting training...")
    for episode in range(episodes):
        state = agent.env.reset()  # Use reset() to get the full state including occupancy
        total_reward = 0
        episode_energy = 0
        
        while True:
            action = agent.choose_action(state)
            next_state, reward = agent.env.step(action)
            total_reward += reward
            
            # Track energy usage
            if action in [1, 2]:  # Heating or cooling
                episode_energy += 1
            
            # Store experience for DQN
            if isinstance(agent, DQNAgent):
                agent.store_experience(state, action, reward, next_state)
                agent.update_networks()
            else:
                agent.update_q_table(state, action, reward, next_state)
            
            state = next_state
            
            # End episode if we've gone through a full day
            if agent.env.hour == 0 and agent.env.day == 0:
                break
        
        # Decay exploration rate
        agent.exploration_rate *= agent.exploration_decay
        total_rewards.append(total_reward)
        exploration_rates.append(agent.exploration_rate)
        energy_usage_per_episode.append(episode_energy)
        
        if episode % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            avg_energy = np.mean(energy_usage_per_episode[-100:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, "
                  f"Exploration Rate: {agent.exploration_rate:.3f}, "
                  f"Average Energy: {avg_energy:.2f}")
    
    return total_rewards, exploration_rates, energy_usage_per_episode

def test_agent(agent, test_episodes=5):
    """Test the trained agent and return test data"""
    print("\nTesting the trained agent...")
    test_episodes_data = []
    
    for episode in range(test_episodes):
        state = (agent.env.temperature, agent.env.day, agent.env.hour, 
                agent.env.weather.get_temperature(agent.env.hour, agent.env.day))
        total_reward = 0
        episode_data = {'steps': []}
        
        print(f"\nTest Episode {episode + 1}:")
        print(f"Initial state - Temperature: {state[0]}°C, Day: {state[1]}, "
              f"Hour: {state[2]}, External Temp: {state[3]:.1f}°C")
        
        while True:
            action = agent.get_optimal_action(state)
            next_state, reward = agent.env.step(action)
            total_reward += reward
            
            # Store step data
            episode_data['steps'].append({
                'temperature': next_state[0],
                'external_temp': next_state[3],
                'action': action,
                'reward': reward
            })
            
            print(f"Action: {['Nothing', 'Heating', 'Cooling'][action]}, "
                  f"Temperature: {next_state[0]}°C, "
                  f"External: {next_state[3]:.1f}°C, "
                  f"Reward: {reward}")
            
            state = next_state
            
            if agent.env.hour == 0 and agent.env.day == 0:
                break
        
        test_episodes_data.append(episode_data)
        print(f"Test Episode {episode + 1} Total Reward: {total_reward}")
    
    return test_episodes_data

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and test a thermostat control agent')
    parser.add_argument('--agent', type=str, choices=['qlearning', 'dqn'], 
                      default='qlearning', help='Type of agent to use (qlearning or dqn)')
    parser.add_argument('--episodes', type=int, default=1000,
                      help='Number of training episodes')
    args = parser.parse_args()
    
    # Create the appropriate agent
    if args.agent == 'qlearning':
        print("Using Q-Learning agent")
        agent = QLearningAgent()
    else:
        print("Using Deep Q-Learning (DQN) agent")
        agent = DQNAgent()
    
    # Train the agent
    total_rewards, exploration_rates, energy_usage = train_agent(agent, args.episodes)
    
    print("\nTraining completed!")
    print(f"Final exploration rate: {agent.exploration_rate:.3f}")
    print(f"Final average reward: {np.mean(total_rewards[-100:]):.2f}")
    print(f"Final average energy usage: {np.mean(energy_usage[-100:]):.2f}")
    
    # Plot training results
    plot_training_results(total_rewards, exploration_rates, energy_usage)
    
    # Test the agent
    test_episodes_data = test_agent(agent)
    
    # Plot test results
    plot_test_results(test_episodes_data)
    
    # Save the trained model if it's a DQN agent
    if isinstance(agent, DQNAgent):
        agent.save_model('dqn_model.pth')
        print("\nSaved DQN model to 'dqn_model.pth'")

if __name__ == "__main__":
    main()
