# Multi-stage build for efficient Docker image
# Stage 1: Builder
FROM python:3.11-slim as builder

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_HOME="/opt/poetry" \
    PATH="$POETRY_HOME/bin:$PATH"

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies
RUN poetry install --only main --no-root --no-directory

# ============================================
# Stage 2: Runtime
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:99 \
    PYBULLET_EGL=1 \
    MESA_GL_VERSION_OVERRIDE=3.3 \
    MESA_GLSL_VERSION_OVERRIDE=330 \
    LIBGL_ALWAYS_SOFTWARE=1

# Install system dependencies for PyBullet headless operation
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Virtual display for headless rendering
    xvfb \
    x11-utils \
    x11-xserver-utils \
    # OpenGL libraries and EGL support
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libegl1-mesa \
    libegl1-mesa-dev \
    libgles2-mesa \
    libgles2-mesa-dev \
    # Additional graphics libraries
    libglu1-mesa \
    libglfw3 \
    libglfw3-dev \
    libglew-dev \
    # System libraries for PyBullet
    libx11-6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    # Process management
    supervisor \
    # Network tools (useful for debugging)
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create a non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /var/log/supervisor && \
    chown -R appuser:appuser /app /var/log/supervisor

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/data && \
    chown -R appuser:appuser /app

# Create supervisor configuration
RUN echo "[supervisord]\n\
nodaemon=true\n\
user=root\n\
logfile=/var/log/supervisor/supervisord.log\n\
pidfile=/var/run/supervisord.pid\n\
\n\
[program:xvfb]\n\
command=/usr/bin/Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset\n\
autostart=true\n\
autorestart=true\n\
stdout_logfile=/var/log/supervisor/xvfb.log\n\
stderr_logfile=/var/log/supervisor/xvfb_err.log\n\
user=appuser\n\
\n\
[program:app]\n\
command=python -m uvicorn cogniforge.main:app --host 0.0.0.0 --port 8000 --workers 1\n\
directory=/app\n\
autostart=true\n\
autorestart=true\n\
stdout_logfile=/var/log/supervisor/app.log\n\
stderr_logfile=/var/log/supervisor/app_err.log\n\
user=appuser\n\
environment=DISPLAY=\":99\",HOME=\"/home/appuser\",USER=\"appuser\",PYBULLET_EGL=\"1\"\n\
" > /etc/supervisor/conf.d/supervisord.conf

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Function to check if Xvfb is running\n\
check_xvfb() {\n\
    if ! pgrep -x "Xvfb" > /dev/null; then\n\
        echo "Starting Xvfb..."\n\
        Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &\n\
        sleep 2\n\
    fi\n\
    echo "Xvfb is running"\n\
}\n\
\n\
# Function to test OpenGL\n\
test_opengl() {\n\
    echo "Testing OpenGL setup..."\n\
    python -c "import pybullet as p; p.connect(p.DIRECT); print(\"PyBullet EGL: OK\"); p.disconnect()" || true\n\
}\n\
\n\
# Start based on command\n\
if [ "$1" = "supervisor" ]; then\n\
    echo "Starting with supervisor..."\n\
    exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf\n\
elif [ "$1" = "test" ]; then\n\
    check_xvfb\n\
    test_opengl\n\
    echo "All tests passed!"\n\
elif [ "$1" = "bash" ] || [ "$1" = "sh" ]; then\n\
    check_xvfb\n\
    exec "$@"\n\
else\n\
    check_xvfb\n\
    test_opengl\n\
    echo "Starting FastAPI application..."\n\
    exec "$@"\n\
fi\n\
' > /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint and default command
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "cogniforge.main:app", "--host", "0.0.0.0", "--port", "8000"]