import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from agent import QLearningAgent
from dqn_agent import DQNAgent
import threading
import queue

class AgentComparisonGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Thermostat Control Agents Comparison")
        self.root.geometry("1200x800")
        
        # Create agents
        self.q_agent = QLearningAgent()
        self.dqn_agent = DQNAgent()
        
        # Training state
        self.is_training = False
        self.current_agent = None
        self.training_queue = queue.Queue()
        
        # Create main frames
        self.create_control_frame()
        self.create_metrics_frame()
        self.create_visualization_frame()
        
        # Initialize metrics storage
        self.q_metrics = {
            'rewards': [],
            'energy': [],
            'exploration': []
        }
        self.dqn_metrics = {
            'rewards': [],
            'energy': [],
            'exploration': []
        }
        
        # Create plots
        self.create_plots()
        
        # Start update loop
        self.update_plots()
    
    def create_control_frame(self):
        """Create the control panel frame"""
        control_frame = ttk.LabelFrame(self.root, text="Control Panel", padding="5")
        control_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Training controls
        ttk.Label(control_frame, text="Training Episodes:").grid(row=0, column=0, padx=5, pady=5)
        self.episodes_var = tk.StringVar(value="1000")
        ttk.Entry(control_frame, textvariable=self.episodes_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # Training buttons for each agent
        self.q_train_button = ttk.Button(control_frame, text="Train Q-Learning", 
                                       command=lambda: self.start_training(self.q_agent, True))
        self.q_train_button.grid(row=1, column=0, padx=5, pady=5)
        
        self.dqn_train_button = ttk.Button(control_frame, text="Train DQN", 
                                         command=lambda: self.start_training(self.dqn_agent, False))
        self.dqn_train_button.grid(row=1, column=1, padx=5, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var).grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        
        # Test buttons
        ttk.Button(control_frame, text="Test Q-Learning", 
                  command=lambda: self.test_agent(self.q_agent)).grid(row=4, column=0, padx=5, pady=5)
        ttk.Button(control_frame, text="Test DQN", 
                  command=lambda: self.test_agent(self.dqn_agent)).grid(row=4, column=1, padx=5, pady=5)
        
        # Save models button
        ttk.Button(control_frame, text="Save Models", 
                  command=self.save_models).grid(row=5, column=0, columnspan=2, padx=5, pady=5)
    
    def create_metrics_frame(self):
        """Create the metrics display frame"""
        metrics_frame = ttk.LabelFrame(self.root, text="Current Metrics", padding="5")
        metrics_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Q-Learning metrics
        ttk.Label(metrics_frame, text="Q-Learning:").grid(row=0, column=0, padx=5, pady=5)
        self.q_reward_var = tk.StringVar(value="Reward: 0.0")
        self.q_energy_var = tk.StringVar(value="Energy: 0.0")
        self.q_explore_var = tk.StringVar(value="Exploration: 1.0")
        ttk.Label(metrics_frame, textvariable=self.q_reward_var).grid(row=1, column=0, padx=5, pady=2)
        ttk.Label(metrics_frame, textvariable=self.q_energy_var).grid(row=2, column=0, padx=5, pady=2)
        ttk.Label(metrics_frame, textvariable=self.q_explore_var).grid(row=3, column=0, padx=5, pady=2)
        
        # DQN metrics
        ttk.Label(metrics_frame, text="DQN:").grid(row=0, column=1, padx=5, pady=5)
        self.dqn_reward_var = tk.StringVar(value="Reward: 0.0")
        self.dqn_energy_var = tk.StringVar(value="Energy: 0.0")
        self.dqn_explore_var = tk.StringVar(value="Exploration: 1.0")
        ttk.Label(metrics_frame, textvariable=self.dqn_reward_var).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(metrics_frame, textvariable=self.dqn_energy_var).grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(metrics_frame, textvariable=self.dqn_explore_var).grid(row=3, column=1, padx=5, pady=2)
    
    def create_visualization_frame(self):
        """Create the visualization frame"""
        viz_frame = ttk.LabelFrame(self.root, text="Training Progress", padding="5")
        viz_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        # Create figure for plots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(12, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_plots(self):
        """Initialize the plots"""
        # Rewards plot
        self.ax1.set_title("Rewards")
        self.ax1.set_xlabel("Episode")
        self.ax1.set_ylabel("Reward")
        self.q_reward_line, = self.ax1.plot([], [], label='Q-Learning', color='blue')
        self.dqn_reward_line, = self.ax1.plot([], [], label='DQN', color='red')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Energy usage plot
        self.ax2.set_title("Energy Usage")
        self.ax2.set_xlabel("Episode")
        self.ax2.set_ylabel("Energy")
        self.q_energy_line, = self.ax2.plot([], [], label='Q-Learning', color='blue')
        self.dqn_energy_line, = self.ax2.plot([], [], label='DQN', color='red')
        self.ax2.legend()
        self.ax2.grid(True)
        
        # Exploration rate plot
        self.ax3.set_title("Exploration Rate")
        self.ax3.set_xlabel("Episode")
        self.ax3.set_ylabel("Rate")
        self.q_explore_line, = self.ax3.plot([], [], label='Q-Learning', color='blue')
        self.dqn_explore_line, = self.ax3.plot([], [], label='DQN', color='red')
        self.ax3.legend()
        self.ax3.grid(True)
        
        self.fig.tight_layout()
    
    def update_plots(self):
        """Update the plots with new data"""
        # Update Q-Learning plots
        self.q_reward_line.set_data(range(len(self.q_metrics['rewards'])), self.q_metrics['rewards'])
        self.q_energy_line.set_data(range(len(self.q_metrics['energy'])), self.q_metrics['energy'])
        self.q_explore_line.set_data(range(len(self.q_metrics['exploration'])), self.q_metrics['exploration'])
        
        # Update DQN plots
        self.dqn_reward_line.set_data(range(len(self.dqn_metrics['rewards'])), self.dqn_metrics['rewards'])
        self.dqn_energy_line.set_data(range(len(self.dqn_metrics['energy'])), self.dqn_metrics['energy'])
        self.dqn_explore_line.set_data(range(len(self.dqn_metrics['exploration'])), self.dqn_metrics['exploration'])
        
        # Update axes limits
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.relim()
            ax.autoscale_view()
        
        self.canvas.draw()
        
        # Schedule next update
        self.root.after(1000, self.update_plots)
    
    def train_agent(self, agent, episodes, is_q_learning=True):
        """Train an agent and update metrics"""
        metrics = self.q_metrics if is_q_learning else self.dqn_metrics
        agent_name = "Q-Learning" if is_q_learning else "DQN"
        
        for episode in range(episodes):
            if not self.is_training:
                break
                
            state = agent.env.reset()  # Use reset() to get the full state including occupancy
            total_reward = 0
            episode_energy = 0
            
            while True:
                action = agent.choose_action(state)
                next_state, reward = agent.env.step(action)
                total_reward += reward
                
                if action in [1, 2]:  # Heating or cooling
                    episode_energy += 1
                
                if isinstance(agent, DQNAgent):
                    agent.store_experience(state, action, reward, next_state)
                    agent.update_networks()
                else:
                    agent.update_q_table(state, action, reward, next_state)
                
                state = next_state
                
                if agent.env.hour == 0 and agent.env.day == 0:
                    break
            
            # Update metrics
            agent.exploration_rate *= agent.exploration_decay
            metrics['rewards'].append(total_reward)
            metrics['energy'].append(episode_energy)
            metrics['exploration'].append(agent.exploration_rate)
            
            # Update progress
            progress = (episode + 1) / episodes * 100
            self.progress_var.set(progress)
            self.status_var.set(f"Training {agent_name} - Episode {episode + 1}/{episodes}")
            
            # Put metrics in queue for GUI update
            self.training_queue.put({
                'rewards': total_reward,
                'energy': episode_energy,
                'exploration': agent.exploration_rate
            })
    
    def start_training(self, agent, is_q_learning=True):
        """Start training a specific agent"""
        if self.is_training:
            self.is_training = False
            if is_q_learning:
                self.q_train_button.config(text="Train Q-Learning")
            else:
                self.dqn_train_button.config(text="Train DQN")
            self.status_var.set("Training stopped")
            return
        
        self.is_training = True
        agent_name = "Q-Learning" if is_q_learning else "DQN"
        if is_q_learning:
            self.q_train_button.config(text="Stop Training")
        else:
            self.dqn_train_button.config(text="Stop Training")
        
        # Clear previous metrics for this agent
        metrics = self.q_metrics if is_q_learning else self.dqn_metrics
        for key in metrics:
            metrics[key] = []
        
        # Get number of episodes
        try:
            episodes = int(self.episodes_var.get())
        except ValueError:
            episodes = 1000
        
        # Start training thread
        def training_thread():
            self.status_var.set(f"Training {agent_name}...")
            self.train_agent(agent, episodes, is_q_learning)
            
            if self.is_training:
                self.status_var.set(f"{agent_name} training completed")
                self.is_training = False
                if is_q_learning:
                    self.q_train_button.config(text="Train Q-Learning")
                else:
                    self.dqn_train_button.config(text="Train DQN")
        
        threading.Thread(target=training_thread, daemon=True).start()
    
    def test_agent(self, agent, test_episodes=5, text_widget=None):
        """Test the trained agent and display results"""
        # Create test results window if no text widget provided
        if text_widget is None:
            test_window = tk.Toplevel(self.root)
            test_window.title(f"{'Q-Learning' if isinstance(agent, QLearningAgent) else 'DQN'} Test Results")
            test_window.geometry("600x400")
            
            # Create text widget with scrollbar
            frame = ttk.Frame(test_window)
            frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            scrollbar = ttk.Scrollbar(frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            text_widget = tk.Text(frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=text_widget.yview)
            
            # Add close button
            ttk.Button(test_window, text="Close", command=test_window.destroy).pack(pady=5)
        
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        
        for episode in range(test_episodes):
            state = agent.env.reset()  # Use reset() to get the full state including occupancy
            total_reward = 0
            
            text_widget.insert(tk.END, f"Test Episode {episode + 1}:\n")
            text_widget.insert(tk.END, f"Initial state - Temperature: {state[0]}°C, Day: {state[1]}, "
                                     f"Hour: {state[2]}, External Temp: {state[3]:.1f}°C, "
                                     f"Occupancy: {state[4]:.2f}\n")
            
            while True:
                action = agent.get_optimal_action(state)
                next_state, reward = agent.env.step(action)
                total_reward += reward
                
                text_widget.insert(tk.END, f"Action: {['Nothing', 'Heating', 'Cooling'][action]}, "
                                         f"Temperature: {next_state[0]}°C, "
                                         f"External: {next_state[3]:.1f}°C, "
                                         f"Occupancy: {next_state[4]:.2f}, "
                                         f"Reward: {reward}\n")
                
                state = next_state
                
                if agent.env.hour == 0 and agent.env.day == 0:
                    break
            
            text_widget.insert(tk.END, f"Test Episode {episode + 1} Total Reward: {total_reward}\n\n")
        
        # Make text widget read-only
        text_widget.config(state=tk.DISABLED)
    
    def save_models(self):
        """Save both agents' models"""
        if isinstance(self.dqn_agent, DQNAgent):
            self.dqn_agent.save_model('dqn_model.pth')
            print("Saved DQN model to 'dqn_model.pth'")

def main():
    root = tk.Tk()
    app = AgentComparisonGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 