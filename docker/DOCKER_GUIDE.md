# Docker Setup Guide for CogniForge

## 📋 Overview

This Docker setup provides a CPU-optimized container for running CogniForge with PyBullet in headless mode using Xvfb and EGL rendering.

## 🚀 Features

- **Headless PyBullet**: Runs physics simulations without GUI using Xvfb virtual display
- **EGL Rendering**: Hardware-accelerated OpenGL rendering via EGL
- **Multi-stage Build**: Optimized image size using multi-stage Docker build
- **Non-root User**: Runs as unprivileged user for security
- **Health Checks**: Built-in health monitoring
- **Supervisor**: Process management for Xvfb and FastAPI
- **Volume Mounts**: Persistent storage for models, data, and logs

## 🛠 Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Docker Compose (optional, for easier management)
- At least 4GB of available RAM
- 10GB of free disk space

## 📦 Quick Start

### Windows (PowerShell)

```powershell
# Navigate to project root
cd C:\Users\Dhenenjay\cogniforge

# Build the image
.\docker\docker-run.ps1 build

# Run the container
.\docker\docker-run.ps1 run

# Or use docker-compose
.\docker\docker-run.ps1 compose-up
```

### Linux/Mac (Bash)

```bash
# Navigate to project root
cd /path/to/cogniforge

# Make script executable
chmod +x docker/docker-run.sh

# Build the image
./docker/docker-run.sh build

# Run the container
./docker/docker-run.sh run

# Or use docker-compose
./docker/docker-run.sh compose-up
```

## 🔧 Manual Docker Commands

### Build Image

```bash
docker build -t cogniforge:latest -f docker/Dockerfile .
```

### Run Container

```bash
docker run -d \
  --name cogniforge-app \
  -p 8000:8000 \
  -e DISPLAY=:99 \
  -e PYBULLET_EGL=1 \
  --env-file .env \
  cogniforge:latest
```

### Run with Development Volumes

```bash
docker run -d \
  --name cogniforge-app \
  -p 8000:8000 \
  -e DISPLAY=:99 \
  -e PYBULLET_EGL=1 \
  -v $(pwd)/cogniforge:/app/cogniforge:ro \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  cogniforge:latest
```

## 🐳 Docker Compose

### Start Services

```bash
docker-compose -f docker/docker-compose.yml up -d
```

### Stop Services

```bash
docker-compose -f docker/docker-compose.yml down
```

### View Logs

```bash
docker-compose -f docker/docker-compose.yml logs -f
```

### Scale Services

```bash
docker-compose -f docker/docker-compose.yml up -d --scale cogniforge=3
```

## 🔍 Testing PyBullet Headless

### Test EGL Rendering

```bash
docker run --rm \
  -e DISPLAY=:99 \
  -e PYBULLET_EGL=1 \
  cogniforge:latest test
```

### Interactive Testing

```bash
# Open shell in container
docker exec -it cogniforge-app bash

# Inside container, test PyBullet
python -c "import pybullet as p; p.connect(p.DIRECT); print('PyBullet OK'); p.disconnect()"
```

## 📁 Directory Structure

```
cogniforge/
├── docker/
│   ├── Dockerfile           # Main Docker image definition
│   ├── docker-compose.yml   # Docker Compose configuration
│   ├── docker-run.sh        # Unix/Linux helper script
│   ├── docker-run.ps1       # Windows PowerShell helper script
│   └── DOCKER_GUIDE.md      # This file
├── .dockerignore            # Files to exclude from Docker build
├── models/                  # ML models (mounted as volume)
├── data/                    # Application data (mounted as volume)
└── logs/                    # Application logs (mounted as volume)
```

## 🔐 Environment Variables

Key environment variables for PyBullet headless operation:

| Variable | Value | Description |
|----------|-------|-------------|
| `DISPLAY` | `:99` | Virtual display number |
| `PYBULLET_EGL` | `1` | Enable EGL rendering |
| `MESA_GL_VERSION_OVERRIDE` | `3.3` | OpenGL version |
| `MESA_GLSL_VERSION_OVERRIDE` | `330` | GLSL version |
| `LIBGL_ALWAYS_SOFTWARE` | `1` | Force software rendering |

## 🏗 Architecture Details

### Multi-stage Build

1. **Builder Stage**: Installs Poetry and Python dependencies
2. **Runtime Stage**: Minimal image with only runtime requirements

### System Packages

- **Xvfb**: Virtual framebuffer for headless display
- **Mesa/EGL**: OpenGL implementation and EGL support
- **Supervisor**: Process management
- **OpenGL Libraries**: Required for PyBullet rendering

### Process Management

Supervisor manages two processes:
1. **Xvfb**: Virtual display server
2. **Uvicorn**: FastAPI application server

## 🔨 Customization

### Modify Display Resolution

Edit the Xvfb command in Dockerfile:
```bash
# Change from 1920x1080 to 1280x720
Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset
```

### Adjust Resource Limits

Edit docker-compose.yml:
```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'  # Increase CPU limit
      memory: 8G   # Increase memory limit
```

### Add GPU Support (NVIDIA)

For NVIDIA GPU support, modify Dockerfile:
```dockerfile
# Add NVIDIA runtime
FROM nvidia/cuda:11.8.0-base-ubuntu22.04
# Install CUDA-enabled PyTorch
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 🐞 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs cogniforge-app

# Check if port is in use
netstat -an | grep 8000
```

### PyBullet EGL Errors

```bash
# Test EGL setup
docker exec cogniforge-app python -c "import os; print(os.environ.get('PYBULLET_EGL'))"

# Check OpenGL libraries
docker exec cogniforge-app ldconfig -p | grep -E "GL|EGL"
```

### Permission Issues

```bash
# Fix volume permissions
docker exec -u root cogniforge-app chown -R appuser:appuser /app
```

### Out of Memory

```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory
# Linux: Update /etc/docker/daemon.json
```

## 📊 Performance Optimization

### Build Cache

```bash
# Use BuildKit for better caching
DOCKER_BUILDKIT=1 docker build -t cogniforge:latest -f docker/Dockerfile .
```

### Layer Caching

```bash
# Build with cache from registry
docker build \
  --cache-from cogniforge:latest \
  -t cogniforge:latest \
  -f docker/Dockerfile .
```

### Slim Image

Current image optimizations:
- Python slim base image
- Multi-stage build
- Minimal package installation
- Cleaned apt cache

## 🚢 Production Deployment

### Security Hardening

1. Use read-only root filesystem
2. Run as non-root user (already configured)
3. Limit capabilities
4. Use secrets management for API keys

### Health Monitoring

```yaml
# Enhanced health check
healthcheck:
  test: ["CMD-SHELL", "curl -f http://localhost:8000/health && pgrep Xvfb"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### Logging

```yaml
# Configure logging driver
logging:
  driver: "json-file"
  options:
    max-size: "100m"
    max-file: "10"
    compress: "true"
```

## 📚 Additional Resources

- [PyBullet Documentation](https://pybullet.org/)
- [EGL Rendering Guide](https://www.khronos.org/egl)
- [Xvfb Documentation](https://www.x.org/releases/X11R7.6/doc/man/man1/Xvfb.1.xhtml)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

## 🤝 Support

For issues related to:
- **PyBullet**: Check EGL and display settings
- **FastAPI**: Review application logs in `/app/logs`
- **Docker**: Ensure Docker daemon is running and has sufficient resources
- **Networking**: Verify port mappings and firewall settings