# 🤖 CogniForge - Adaptive RL Environment for Robotic Learning

**Adaptive RL Environment to train robots with natural language**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-success.svg)](https://github.com/yourusername/cogniforge)

CogniForge is a comprehensive platform that enables robots to learn complex manipulation tasks through natural language descriptions. It combines behavioral cloning, reinforcement learning optimization, and vision-based corrections to create production-ready robotic policies.

## ✨ Key Features

- **🗣️ Natural Language Interface**: Describe robotic tasks in plain English
- **🧠 Behavioral Cloning Pipeline**: Learn from expert demonstrations with neural networks  
- **⚙️ Advanced Optimization**: CMA-ES and PPO algorithms for policy refinement
- **👁️ Vision Integration**: Real-time visual corrections and feedback
- **💻 Automatic Code Generation**: Production-ready robot control code
- **📊 Real-time Monitoring**: Live metrics, training curves, and execution tracking
- **🎮 PyBullet Simulation**: High-fidelity physics-based robot simulation
- **🌐 Modern Web Interface**: Responsive dashboard with real-time updates

## 🏗️ System Architecture

CogniForge implements a sophisticated multi-stage pipeline:

```
Natural Language Input → Planning Engine → Expert Demonstration
                                           ↓
Production Code ← Vision Refinement ← RL Optimization ← Behavioral Cloning
```

### Core Components:

1. **📋 Planning Engine**: Converts natural language to structured robotic tasks
2. **👨‍🏫 Expert Demonstration**: Generates high-quality training trajectories
3. **🧠 Behavioral Cloning**: Neural network training on expert data
4. **⚙️ RL Optimization**: Policy refinement using CMA-ES/PPO algorithms
5. **👁️ Vision System**: Real-time visual corrections and offset detection
6. **💻 Code Generator**: Automatic production code synthesis
7. **🌐 Web Dashboard**: Real-time monitoring and control interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyBullet physics engine
- Modern web browser for the dashboard

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cogniforge.git
cd cogniforge

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
```

### Basic Usage

#### 1. Start the Backend API

```bash
python main.py
```

This starts the FastAPI backend server on `http://localhost:8000`.

#### 2. Launch the Web Interface

```bash
# Navigate to frontend directory
cd frontend

# Start the frontend server
python -m http.server 8080
```

Access the web interface at `http://localhost:8080`.

#### 3. Execute Tasks

1. Open the web interface
2. Enter a natural language task description:
   - "Pick up the blue cube and place it on the red platform"
   - "Move the yellow block to the corner"
   - "Stack the cubes in ascending order"
3. Configure options:
   - **Use Vision**: Enable real-time visual corrections
   - **GPT Rewards**: Use GPT for reward function generation
   - **Dry Run**: Test without actual execution
   - **BC Epochs**: Number of behavioral cloning training epochs
   - **Opt Steps**: Optimization algorithm iterations
4. Click **Execute** to start the pipeline

## 📊 Pipeline Stages

### 1. Planning Phase
- Parses natural language input
- Generates task structure and constraints
- Defines success criteria

### 2. Expert Demonstration
- Creates initial trajectory demonstrations
- Generates training data for behavioral cloning
- Uses motion planning algorithms

### 3. Behavioral Cloning Training
- Trains neural networks on expert demonstrations
- Real-time loss tracking and visualization
- Configurable epochs and batch sizes

### 4. RL Optimization
- Refines policies using CMA-ES or PPO
- Population-based optimization
- Live reward tracking and convergence monitoring

### 5. Vision Refinement
- Real-time visual feedback integration
- Object detection and pose estimation
- GPT-powered vision corrections

### 6. Code Generation
- Automatic production code synthesis
- Framework-agnostic output (PyBullet, ROS, etc.)
- Full documentation and testing

### 7. Execution
- Deploys optimized policies
- Real-time monitoring and metrics
- Success validation

## 🎛️ Configuration Options

### Web Interface Controls

- **Use Vision**: Toggle vision-based corrections
- **GPT Rewards**: Enable GPT-powered reward functions
- **Dry Run**: Test pipeline without robot execution
- **BC Epochs**: Configure behavioral cloning training duration
- **Opt Steps**: Set reinforcement learning optimization iterations

### API Configuration

```python
# Example API usage
import requests

response = requests.post('http://localhost:8000/execute', json={
    "command": "Pick up the blue cube and place it on the platform",
    "use_vision": True,
    "use_gpt_reward": False,
    "dry_run": True,
    "bc_epochs": 10,
    "optimization_steps": 50
})

request_id = response.json()['request_id']
```

### Environment Variables

```bash
# .env file configuration
OPENAI_API_KEY=your_openai_api_key_here  # Optional, for GPT features
PYBULLET_RENDER=1                        # Enable PyBullet GUI
LOG_LEVEL=INFO                           # Logging verbosity
```

## 🔄 Real-time Monitoring

The web interface provides comprehensive real-time monitoring:

### Live Metrics
- **Progress tracking** with phase indicators
- **Training curves** for loss and reward
- **Execution time** and performance metrics
- **Success rates** and completion statistics

### Console Output
- Real-time log streaming
- Color-coded message types (info, warning, error)
- Detailed training progress
- API call traces and responses

### Visualization
- Phase progression indicators
- Interactive charts for metrics
- Real-time data updates via Server-Sent Events

## 📁 Project Structure

```
cogniforge/
├── cogniforge/                 # Core package
│   ├── __init__.py
│   ├── cli.py                 # Command-line interface
│   ├── controllers/           # Robot controllers
│   ├── learning/             # ML algorithms
│   ├── optimization/         # RL and optimization
│   ├── planning/            # Task planning
│   ├── ui/                  # UI components  
│   └── vision/              # Vision systems
├── examples/                # Example scripts
├── frontend/               # Web interface
│   ├── index.html         # Main dashboard
│   └── assets/           # Static resources
├── tests/                # Test suite
├── main.py              # FastAPI backend
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## 🧪 Examples

### Command Line Interface

```bash
# Run a simple pick and place task
python -m cogniforge.cli execute "Pick up the red cube"

# Use with specific parameters
python -m cogniforge.cli execute "Stack the blocks" \
    --bc-epochs 20 \
    --optimization-steps 100 \
    --use-vision \
    --dry-run
```

### Python API

```python
from cogniforge import CogniForgeAgent

# Initialize agent
agent = CogniForgeAgent()

# Execute task
result = agent.execute_task(
    command="Pick up the blue cube and place it on the platform",
    use_vision=True,
    bc_epochs=15,
    optimization_steps=50
)

print(f"Task completed: {result.success}")
print(f"Execution time: {result.duration}s")
```

## 🔧 API Reference

### Main Endpoints

- `POST /execute` - Start task execution
- `GET /stream/{request_id}` - Real-time updates via SSE
- `GET /status/{request_id}` - Check execution status
- `GET /results/{request_id}` - Get execution results
- `GET /health` - System health check

### Request/Response Examples

```json
// POST /execute
{
  "command": "Pick up the blue cube and place it on the platform",
  "use_vision": true,
  "use_gpt_reward": false,
  "dry_run": false,
  "bc_epochs": 10,
  "optimization_steps": 20
}

// Response
{
  "request_id": "req_123456789",
  "status": "started",
  "message": "Task execution initiated"
}
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_learning.py      # ML components
pytest tests/test_vision.py       # Vision systems
pytest tests/test_integration.py  # End-to-end tests

# Run with coverage
pytest --cov=cogniforge tests/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyBullet team for the physics simulation engine
- OpenAI for GPT integration capabilities
- The robotics community for inspiration and feedback

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/cogniforge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/cogniforge/discussions)
- **Email**: support@cogniforge.ai

---

**CogniForge** - Transforming natural language into robotic intelligence. 🤖✨