import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from env import ThermostatEnv
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class DQNNetwork(nn.Module):
    """
    Neural Network for Deep Q-Learning.
    Takes state as input and outputs Q-values for each action.
    """
    def __init__(self, input_size, output_size):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),  # Increased network capacity
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """
    Deep Q-Learning Agent for the Thermostat Environment.
    
    This agent uses a neural network to approximate the Q-function instead of a Q-table.
    Key features:
    - Experience replay buffer to store and sample past experiences
    - Target network for stable learning
    - Epsilon-greedy exploration strategy
    - Neural network with 3 hidden layers
    
    The state space is continuous, so we don't need to discretize it like in Q-learning.
    """
    def __init__(self, learning_rate=0.001, discount_factor=0.95, 
                 exploration_rate=1.0, exploration_decay=0.995,
                 memory_size=10000, batch_size=64, target_update=10):
        self.env = ThermostatEnv()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Initialize TensorBoard writer
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.writer = SummaryWriter(f'runs/dqn_thermostat_{current_time}')
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Initialize networks
        self.input_size = 5  # temperature, day, hour, external_temp, occupancy
        self.output_size = 3  # number of actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQNNetwork(self.input_size, self.output_size).to(self.device)
        self.target_net = DQNNetwork(self.input_size, self.output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.steps = 0
        self.episode_rewards = []
        self.episode_energy = []
    
    def log_metrics(self, episode, loss=None, reward=None, energy=None):
        """Log metrics to TensorBoard"""
        if loss is not None:
            self.writer.add_scalar('Training/Loss', loss, episode)
        if reward is not None:
            self.writer.add_scalar('Training/Reward', reward, episode)
        if energy is not None:
            self.writer.add_scalar('Training/Energy_Consumption', energy, episode)
        
        self.writer.add_scalar('Training/Exploration_Rate', self.exploration_rate, episode)
        
        # Log network parameters
        for name, param in self.policy_net.named_parameters():
            self.writer.add_histogram(f'Network/{name}', param.data.cpu().numpy(), episode)
    
    def preprocess_state(self, state):
        """Convert state tuple to tensor"""
        return torch.FloatTensor(state).to(self.device)
    
    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        With probability epsilon: choose random action
        With probability 1-epsilon: choose best action from policy network
        """
        if random.random() < self.exploration_rate:
            return random.randint(0, 2)  # Random action
        
        with torch.no_grad():
            state_tensor = self.preprocess_state(state)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state))
    
    def update_networks(self):
        """Update policy network using experience replay"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert to numpy arrays first, then to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        
        # Get current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        
        # Compute target Q values
        target_q_values = rewards + self.discount_factor * next_q_values
        
        # Compute loss and update policy network
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Log Q-values
        with torch.no_grad():
            q_values = self.policy_net(states)
            self.writer.add_histogram('Q-values/Distribution', q_values.cpu().numpy(), self.steps)
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def train_episode(self, max_steps=1000):
        """Train for one episode and return metrics"""
        state = self.env.reset()
        episode_reward = 0
        episode_energy = 0
        loss = 0
        
        for step in range(max_steps):
            action = self.choose_action(state)
            next_state, reward = self.env.step(action)
            
            self.store_experience(state, action, reward, next_state)
            step_loss = self.update_networks()
            if step_loss is not None:
                loss = step_loss
            
            episode_reward += reward
            episode_energy += self.env.energy_used
            
            state = next_state
        
        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
        
        return episode_reward, episode_energy, loss
    
    def get_optimal_action(self, state):
        """Get the best action for a given state using the policy network"""
        with torch.no_grad():
            state_tensor = self.preprocess_state(state)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def save_model(self, path):
        """Save the policy network state"""
        torch.save(self.policy_net.state_dict(), path)
    
    def load_model(self, path):
        """Load the policy network state"""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close() 