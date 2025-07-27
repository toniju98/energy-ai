# Energy-Efficient Thermostat Control with Reinforcement Learning

This project implements and compares two reinforcement learning approaches for controlling a smart thermostat system: Q-Learning and Deep Q-Network (DQN). The goal is to maintain comfortable indoor temperatures while minimizing energy consumption.

## Features

- **Two Learning Approaches**:
  - Q-Learning: Traditional tabular approach
  - Deep Q-Network (DQN): Advanced neural network-based approach

- **Realistic Environment Simulation**:
  - Real-time weather integration via OpenWeatherMap API
  - Occupancy pattern simulation (weekday/weekend patterns)
  - Temperature dynamics with heat transfer modeling
  - Day/night cycle simulation
  - Energy consumption tracking

- **Interactive GUI**:
  - Side-by-side comparison of both approaches
  - Real-time training visualization
  - Performance metrics tracking
  - Interactive testing interface

- **Advanced Training Features**:
  - TensorBoard integration for training monitoring
  - Model checkpointing and saving
  - Comprehensive metrics logging
  - Progress tracking with tqdm

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/energy-ai.git
cd energy-ai
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   Create a `.env` file in the project root with:
```bash
OPENWEATHER_API_KEY=your_api_key_here
LATITUDE=40.7128
LONGITUDE=-74.0060
```

## Usage

### Running the Comparison GUI

To start the comparison interface:
```bash
python comparison_gui.py
```

The GUI provides the following features:

1. **Training Controls**:
   - Set number of training episodes
   - Train Q-Learning agent
   - Train DQN agent
   - Monitor training progress
   - Stop training at any time

2. **Performance Metrics**:
   - Real-time reward tracking
   - Energy consumption monitoring
   - Exploration rate visualization

3. **Testing**:
   - Test each agent independently
   - View detailed action sequences
   - Analyze temperature control decisions

### Command Line Training

For headless training and analysis:
```bash
python main.py --episodes 1000 --agent dqn
```

Available options:
- `--episodes`: Number of training episodes
- `--agent`: Choose between 'qlearning' or 'dqn'
- `--test`: Run test episodes after training

### Standalone Training

For focused DQN training with checkpointing:
```bash
python train.py
```

This will train the DQN agent with TensorBoard logging and automatic checkpoint saving.

## Project Structure

```
energy-ai/
├── agent.py              # Q-Learning implementation
├── dqn_agent.py          # DQN implementation with TensorBoard
├── env.py                # Environment simulation with weather API
├── occupancy.py          # Occupancy pattern simulation
├── comparison_gui.py     # GUI for comparing agents
├── main.py              # Command-line training and analysis
├── train.py             # Standalone DQN training script
├── requirements.txt      # Project dependencies
├── checkpoints/         # Saved model checkpoints
├── runs/               # TensorBoard log files
└── README.md           # This file
```

## Key Components

### Q-Learning Agent (`agent.py`)
- Tabular Q-value storage
- Epsilon-greedy exploration
- Simple state-action mapping
- Real-time training visualization

### DQN Agent (`dqn_agent.py`)
- Neural network architecture with PyTorch
- Experience replay buffer
- Target network for stable learning
- Advanced exploration strategies
- TensorBoard integration for monitoring
- Automatic model checkpointing

### Environment (`env.py`)
- Real-time weather data integration
- Occupancy pattern simulation
- Temperature dynamics with heat transfer
- Energy consumption tracking
- Reward system based on comfort and efficiency

### Occupancy Simulation (`occupancy.py`)
- Realistic weekday/weekend patterns
- Time-based occupancy modeling
- Comfort zone adjustments based on occupancy

## Requirements

- Python 3.7+
- PyTorch 2.0+
- NumPy 1.21+
- Matplotlib 3.4+
- Requests 2.26+
- TensorBoard 2.15+
- tqdm 4.65+
- python-dotenv 0.19+

## Environment Setup

The project uses real weather data from OpenWeatherMap API. To set up:

1. Get a free API key from [OpenWeatherMap](https://openweathermap.org/api)
2. Create a `.env` file with your API key and location coordinates
3. The system will automatically fetch weather data and cache it to minimize API calls

## Training Process

1. **GUI Training**:
   - Set the desired number of episodes in the input field
   - Choose which agent to train (Q-Learning or DQN)
   - Monitor real-time progress and metrics
   - View performance plots and analysis

2. **Command Line Training**:
   - Use `main.py` for comprehensive training and analysis
   - Use `train.py` for focused DQN training with checkpointing
   - TensorBoard logs are automatically generated

3. **Model Checkpoints**:
   - DQN models are automatically saved in `checkpoints/`
   - Training progress is logged to TensorBoard
   - Models can be loaded for continued training

## Testing the Agents

1. **GUI Testing**:
   - After training, click the corresponding "Test" button
   - View detailed action sequences and temperature changes
   - Analyze energy usage and total rewards

2. **Command Line Testing**:
   - Use `main.py --test` to run test episodes
   - View comprehensive test results and plots

## Performance Monitoring

- **TensorBoard**: View training metrics at `http://localhost:6006`
- **Real-time Plots**: Monitor rewards, energy usage, and exploration rates
- **Checkpoint Management**: Automatic model saving and loading
- **Comprehensive Logging**: All metrics are logged for analysis

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a feature branch
3. Submitting a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenWeatherMap API for real-time weather data
- PyTorch team for the deep learning framework
- TensorBoard for training visualization
- The reinforcement learning community for research and insights