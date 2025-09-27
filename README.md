# CogniForge

A Python 3.11 project integrating FastAPI, PyBullet physics simulation, and machine learning libraries.

## 🚀 Features

- **FastAPI** web framework for building APIs
- **PyBullet** for physics simulation
- **PyTorch** for deep learning
- **Stable Baselines3** for reinforcement learning
- **Gymnasium** for RL environments
- **CMA-ES** for optimization
- **NumPy** for numerical computing
- **Pillow** for image processing

## 📋 Requirements

- Python 3.11 or higher
- Poetry (for dependency management)

## 🔧 Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd cogniforge
```

2. Install Poetry if you haven't already:
```bash
pip install poetry
```

3. Install dependencies:
```bash
poetry install
```

4. Activate the virtual environment:
```bash
poetry shell
```

## 🏃 Running the Application

### Development Server

Run the FastAPI application with auto-reload:

```bash
poetry run uvicorn cogniforge.main:app --reload
```

Or using Python directly:

```bash
poetry run python -m cogniforge.main
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, you can access:
- Interactive API docs (Swagger UI): `http://localhost:8000/docs`
- Alternative API docs (ReDoc): `http://localhost:8000/redoc`

## 📁 Project Structure

```
cogniforge/
├── cogniforge/          # Main package directory
│   ├── __init__.py     # Package initialization
│   └── main.py         # FastAPI application
├── tests/              # Test directory
├── .env.example        # Example environment variables
├── pyproject.toml      # Poetry configuration and dependencies
└── README.md           # This file
```

## 🔌 API Endpoints

- `GET /` - Root endpoint, returns welcome message
- `GET /health` - Health check endpoint
- `GET /info` - Information about installed libraries and system capabilities

## 🧪 Development

### Running Tests

```bash
poetry run pytest
```

### Code Formatting

```bash
poetry run black cogniforge/
poetry run ruff check cogniforge/
```

### Type Checking

```bash
poetry run mypy cogniforge/
```

## 🔐 Environment Variables

Copy `.env.example` to `.env` and configure your environment variables:

```bash
cp .env.example .env
```

## 📦 Key Dependencies

- **fastapi**: Modern web framework for building APIs
- **uvicorn**: ASGI server for FastAPI
- **pybullet**: Physics simulation for robotics, games, and ML
- **numpy**: Fundamental package for scientific computing
- **pillow**: Python Imaging Library
- **gymnasium**: Standard API for reinforcement learning
- **torch**: PyTorch deep learning framework
- **stable-baselines3**: Reliable RL algorithms
- **cmaes**: CMA-ES optimization algorithm
- **python-dotenv**: Load environment variables from .env file

## 📄 License

[Add your license information here]

## 👥 Contributors

[Add contributor information here]