import numpy as np
from env import ThermostatEnv

class QLearningAgent:
    """
    Q-Learning Agent for the Thermostat Environment.
    
    Q-Learning is a model-free reinforcement learning algorithm that learns the value of an action
    in a particular state. It does not require a model of the environment and can handle problems
    with stochastic transitions and rewards.
    
    The Q-learning algorithm works as follows:
    1. Initialize Q-table with zeros (or random values)
    2. For each episode:
        a. Observe current state s
        b. Choose action a using epsilon-greedy strategy
        c. Take action a, observe reward r and next state s'
        d. Update Q-value using the formula:
           Q(s,a) = Q(s,a) + α[r + γ max(Q(s',a')) - Q(s,a)]
           where:
           - α (alpha) is the learning rate
           - γ (gamma) is the discount factor
           - r is the immediate reward
           - max(Q(s',a')) is the maximum Q-value for the next state
    3. Repeat until convergence or maximum episodes reached
    
    The Q-table dimensions represent:
    - Temperature (15-30°C): 16 possible values
    - Day of week (0-6): 7 possible values
    - Hour of day (0-23): 24 possible values
    - External temperature (discretized): 16 possible values
    - Occupancy (discretized): 5 possible values (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
    - Actions (0-2): 3 possible actions (nothing, heating, cooling)
    
    The Q-value Q(s,a) represents the expected future reward for taking action a in state s.
    The agent learns to maximize the sum of immediate and future rewards.
    """
    def __init__(self, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        """
        Initialize the Q-Learning agent with hyperparameters:
        
        Args:
            learning_rate (float): How quickly the agent updates its Q-values (0-1)
                - Higher values (e.g., 0.5) make the agent learn faster but may be less stable
                - Lower values (e.g., 0.01) make learning slower but more stable
                - Typical range: 0.1 to 0.5
            
            discount_factor (float): How much future rewards are valued (0-1)
                - Higher values (e.g., 0.95) make the agent consider long-term rewards more
                - Lower values (e.g., 0.5) make the agent focus more on immediate rewards
                - Typical range: 0.8 to 0.99
            
            exploration_rate (float): Probability of taking random actions (0-1)
                - Higher values (e.g., 1.0) make the agent explore more
                - Lower values (e.g., 0.1) make the agent exploit learned knowledge more
                - Starts high and decays over time
            
            exploration_decay (float): Rate at which exploration decreases over time
                - Higher values (e.g., 0.999) make exploration decrease slowly
                - Lower values (e.g., 0.9) make exploration decrease quickly
                - Typical range: 0.995 to 0.999
        """
        self.env = ThermostatEnv()
        self.learning_rate = learning_rate  # alpha: controls how much new information overrides old information
        self.discount_factor = discount_factor  # gamma: determines importance of future rewards
        self.exploration_rate = exploration_rate  # epsilon: probability of taking random action
        self.exploration_decay = exploration_decay  # how quickly exploration rate decreases
        
        # Initialize Q-table with zeros
        # Dimensions: (temperature_range, days, hours, external_temp_range, occupancy_range, actions)
        # Temperature range: 15-30°C (16 possible values)
        # Days: 0-6 (7 possible values, Monday-Sunday)
        # Hours: 0-23 (24 possible values)
        # External temperature: 15-30°C (16 possible values)
        # Occupancy: 0-1 (5 possible values: 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
        # Actions: 0 (nothing), 1 (heating), 2 (cooling)
        self.q_table = np.zeros((16, 7, 24, 16, 5, 3))
        
    def get_state_index(self, state):
        """
        Convert the continuous state space into discrete indices for the Q-table.
        
        Args:
            state (tuple): (temperature, day, hour, external_temp, occupancy)
            
        Returns:
            tuple: (temp_index, day, hour, external_temp_index, occupancy_index) for Q-table indexing
        """
        temperature, day, hour, external_temp, occupancy = state
        # Convert temperatures to indices (15-30 -> 0-15)
        temp_index = max(0, min(15, int(temperature - 15)))
        external_temp_index = max(0, min(15, int(external_temp - 15)))
        # Convert occupancy to index (0-1 -> 0-4)
        occupancy_index = min(4, int(occupancy * 5))
        # Ensure day and hour are within bounds
        day = day % 7
        hour = hour % 24
        return (temp_index, day, hour, external_temp_index, occupancy_index)
    
    def choose_action(self, state):
        """
        Epsilon-greedy action selection strategy.
        
        This method implements the exploration-exploitation trade-off:
        - With probability epsilon: choose random action (exploration)
        - With probability 1-epsilon: choose best known action (exploitation)
        
        Exploration is important to:
        1. Discover new strategies
        2. Avoid getting stuck in local optima
        3. Adapt to changes in the environment
        
        Exploitation is important to:
        1. Use learned knowledge effectively
        2. Maximize rewards
        3. Make optimal decisions
        
        Args:
            state (tuple): Current state (temperature, day, hour, external_temp, occupancy)
            
        Returns:
            int: Action to take (0, 1, or 2)
        """
        # Exploration: choose random action
        if np.random.random() < self.exploration_rate:
            return np.random.randint(3)  # Random action
        
        # Exploitation: choose best known action
        state_index = self.get_state_index(state)
        return np.argmax(self.q_table[state_index])
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Update Q-values using the Q-learning update rule:
        Q(s,a) = Q(s,a) + α[r + γ max(Q(s',a')) - Q(s,a)]
        
        This formula updates the Q-value based on:
        1. Current Q-value: Q(s,a)
        2. Learning rate: α
        3. Immediate reward: r
        4. Discounted future reward: γ max(Q(s',a'))
        
        The update rule combines:
        - Current estimate: (1-α)Q(s,a)
        - New information: α[r + γ max(Q(s',a'))]
        
        Where:
        - α (alpha) is the learning rate
        - γ (gamma) is the discount factor
        - r is the reward
        - s' is the next state
        - a' is the next action
        
        Args:
            state (tuple): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (tuple): Next state
        """
        state_index = self.get_state_index(state)
        next_state_index = self.get_state_index(next_state)
        
        # Get current Q-value
        old_value = self.q_table[state_index][action]
        # Get maximum Q-value for next state
        next_max = np.max(self.q_table[next_state_index])
        
        # Q-learning update formula
        new_value = (1 - self.learning_rate) * old_value + \
                    self.learning_rate * (reward + self.discount_factor * next_max)
        
        self.q_table[state_index][action] = new_value
    
    def get_optimal_action(self, state):
        """
        Get the best action for a given state based on learned Q-values.
        Used after training to get optimal policy.
        
        This method implements the greedy policy:
        - Always chooses the action with the highest Q-value
        - No exploration, only exploitation
        - Used for testing and deployment

        Args:
            state (tuple): Current state (temperature, day, hour, external_temp, occupancy)

        Returns:
            int: Best action (0, 1, or 2)
        """
        state_index = self.get_state_index(state)
        return np.argmax(self.q_table[state_index])

    def train(self, episodes=1000, max_steps=1000):
        """
        Train the agent for a specified number of episodes.

        Args:
            episodes (int): Number of episodes to train
            max_steps (int): Maximum steps per episode
        """
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0

            for step in range(max_steps):
                action = self.choose_action(state)
                next_state, reward = self.env.step(action)

                self.update_q_table(state, action, reward, next_state)
                total_reward += reward
                state = next_state

            # Decay exploration rate
            self.exploration_rate *= self.exploration_decay

            # Print progress every 100 episodes
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}")
                print(f"Total Reward: {total_reward:.2f}")
                print(f"Exploration Rate: {self.exploration_rate:.3f}")
                print("---")

if __name__ == "__main__":
    # Create and train the agent
    agent = QLearningAgent()
    agent.train()
