#!/bin/bash

# Docker helper script for CogniForge
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="cogniforge"
CONTAINER_NAME="cogniforge-app"
PORT="${PORT:-8000}"

# Functions
print_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker image"
    echo "  run         Run the container"
    echo "  run-dev     Run with volume mounts for development"
    echo "  test        Test PyBullet headless setup"
    echo "  shell       Open a shell in the container"
    echo "  stop        Stop the container"
    echo "  logs        View container logs"
    echo "  clean       Remove container and image"
    echo "  compose-up  Start using docker-compose"
    echo "  compose-down Stop docker-compose"
    echo ""
}

build_image() {
    echo -e "${GREEN}Building Docker image...${NC}"
    docker build -t ${IMAGE_NAME}:latest -f docker/Dockerfile .
    echo -e "${GREEN}✓ Image built successfully${NC}"
}

run_container() {
    echo -e "${GREEN}Starting container...${NC}"
    docker run -d \
        --name ${CONTAINER_NAME} \
        -p ${PORT}:8000 \
        --env-file .env \
        -e DISPLAY=:99 \
        -e PYBULLET_EGL=1 \
        ${IMAGE_NAME}:latest
    echo -e "${GREEN}✓ Container started on port ${PORT}${NC}"
}

run_dev() {
    echo -e "${GREEN}Starting container in dev mode...${NC}"
    docker run -d \
        --name ${CONTAINER_NAME} \
        -p ${PORT}:8000 \
        --env-file .env \
        -e DISPLAY=:99 \
        -e PYBULLET_EGL=1 \
        -v $(pwd)/cogniforge:/app/cogniforge:ro \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/logs:/app/logs \
        ${IMAGE_NAME}:latest
    echo -e "${GREEN}✓ Container started in dev mode${NC}"
}

test_container() {
    echo -e "${GREEN}Testing PyBullet headless setup...${NC}"
    docker run --rm \
        -e DISPLAY=:99 \
        -e PYBULLET_EGL=1 \
        ${IMAGE_NAME}:latest \
        test
}

open_shell() {
    echo -e "${GREEN}Opening shell in container...${NC}"
    docker exec -it ${CONTAINER_NAME} bash
}

stop_container() {
    echo -e "${YELLOW}Stopping container...${NC}"
    docker stop ${CONTAINER_NAME} || true
    docker rm ${CONTAINER_NAME} || true
    echo -e "${GREEN}✓ Container stopped${NC}"
}

view_logs() {
    echo -e "${GREEN}Container logs:${NC}"
    docker logs -f ${CONTAINER_NAME}
}

clean_all() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME} 2>/dev/null || true
    docker rmi ${IMAGE_NAME}:latest 2>/dev/null || true
    echo -e "${GREEN}✓ Cleanup complete${NC}"
}

compose_up() {
    echo -e "${GREEN}Starting with docker-compose...${NC}"
    docker-compose -f docker/docker-compose.yml up -d
    echo -e "${GREEN}✓ Services started${NC}"
}

compose_down() {
    echo -e "${YELLOW}Stopping docker-compose...${NC}"
    docker-compose -f docker/docker-compose.yml down
    echo -e "${GREEN}✓ Services stopped${NC}"
}

# Main script
case "$1" in
    build)
        build_image
        ;;
    run)
        stop_container
        run_container
        ;;
    run-dev)
        stop_container
        run_dev
        ;;
    test)
        test_container
        ;;
    shell)
        open_shell
        ;;
    stop)
        stop_container
        ;;
    logs)
        view_logs
        ;;
    clean)
        clean_all
        ;;
    compose-up)
        compose_up
        ;;
    compose-down)
        compose_down
        ;;
    *)
        print_help
        exit 1
        ;;
esac