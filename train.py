from dqn_agent import DQNAgent
import numpy as np
from tqdm import tqdm

def train(episodes=1000, max_steps=1000):
    agent = DQNAgent()
    
    try:
        for episode in tqdm(range(episodes), desc="Training"):
            # Train one episode
            reward, energy, loss = agent.train_episode(max_steps)
            
            # Log metrics
            agent.log_metrics(
                episode=episode,
                loss=loss,
                reward=reward,
                energy=energy
            )
            
            # Print progress every 100 episodes
            if (episode + 1) % 100 == 0:
                print(f"\nEpisode {episode + 1}")
                print(f"Average Reward: {reward:.2f}")
                print(f"Energy Consumption: {energy:.2f}")
                print(f"Exploration Rate: {agent.exploration_rate:.3f}")
                
                # Save model checkpoint
                agent.save_model(f"checkpoints/dqn_model_episode_{episode + 1}.pth")
    
    finally:
        # Ensure TensorBoard writer is closed
        agent.close()

if __name__ == "__main__":
    train() 