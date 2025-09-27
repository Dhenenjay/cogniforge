# Docker helper script for CogniForge (Windows/PowerShell)
param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [Parameter()]
    [int]$Port = 8000
)

# Configuration
$ImageName = "cogniforge"
$ContainerName = "cogniforge-app"

# Color functions
function Write-Success {
    Write-Host $args -ForegroundColor Green
}

function Write-Warning {
    Write-Host $args -ForegroundColor Yellow
}

function Write-Error {
    Write-Host $args -ForegroundColor Red
}

# Functions
function Show-Help {
    Write-Host "Usage: .\docker-run.ps1 [COMMAND]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  build       Build the Docker image"
    Write-Host "  run         Run the container"
    Write-Host "  run-dev     Run with volume mounts for development"
    Write-Host "  test        Test PyBullet headless setup"
    Write-Host "  shell       Open a shell in the container"
    Write-Host "  stop        Stop the container"
    Write-Host "  logs        View container logs"
    Write-Host "  clean       Remove container and image"
    Write-Host "  compose-up  Start using docker-compose"
    Write-Host "  compose-down Stop docker-compose"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Port       Specify port (default: 8000)"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\docker-run.ps1 build"
    Write-Host "  .\docker-run.ps1 run -Port 8080"
}

function Build-Image {
    Write-Success "Building Docker image..."
    docker build -t ${ImageName}:latest -f docker/Dockerfile .
    if ($LASTEXITCODE -eq 0) {
        Write-Success "✓ Image built successfully"
    } else {
        Write-Error "✗ Build failed"
        exit 1
    }
}

function Start-Container {
    Write-Success "Starting container..."
    
    # Check if .env file exists
    $envFile = if (Test-Path .env) { "--env-file .env" } else { "" }
    
    $cmd = "docker run -d " +
           "--name $ContainerName " +
           "-p ${Port}:8000 " +
           "$envFile " +
           "-e DISPLAY=:99 " +
           "-e PYBULLET_EGL=1 " +
           "${ImageName}:latest"
    
    Invoke-Expression $cmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "✓ Container started on port $Port"
        Write-Host "Access the API at: http://localhost:$Port"
    } else {
        Write-Error "✗ Failed to start container"
        exit 1
    }
}

function Start-DevContainer {
    Write-Success "Starting container in dev mode..."
    
    # Get current directory
    $CurrentDir = Get-Location
    
    # Check if .env file exists
    $envFile = if (Test-Path .env) { "--env-file .env" } else { "" }
    
    $cmd = "docker run -d " +
           "--name $ContainerName " +
           "-p ${Port}:8000 " +
           "$envFile " +
           "-e DISPLAY=:99 " +
           "-e PYBULLET_EGL=1 " +
           "-v ${CurrentDir}/cogniforge:/app/cogniforge:ro " +
           "-v ${CurrentDir}/models:/app/models " +
           "-v ${CurrentDir}/data:/app/data " +
           "-v ${CurrentDir}/logs:/app/logs " +
           "${ImageName}:latest"
    
    Invoke-Expression $cmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "✓ Container started in dev mode on port $Port"
    } else {
        Write-Error "✗ Failed to start container"
        exit 1
    }
}

function Test-Container {
    Write-Success "Testing PyBullet headless setup..."
    docker run --rm `
        -e DISPLAY=:99 `
        -e PYBULLET_EGL=1 `
        ${ImageName}:latest `
        test
}

function Open-Shell {
    Write-Success "Opening shell in container..."
    docker exec -it $ContainerName bash
}

function Stop-Container {
    Write-Warning "Stopping container..."
    docker stop $ContainerName 2>$null
    docker rm $ContainerName 2>$null
    Write-Success "✓ Container stopped"
}

function Show-Logs {
    Write-Success "Container logs:"
    docker logs -f $ContainerName
}

function Clean-All {
    Write-Warning "Cleaning up..."
    docker stop $ContainerName 2>$null
    docker rm $ContainerName 2>$null
    docker rmi ${ImageName}:latest 2>$null
    Write-Success "✓ Cleanup complete"
}

function Start-Compose {
    Write-Success "Starting with docker-compose..."
    docker-compose -f docker/docker-compose.yml up -d
    if ($LASTEXITCODE -eq 0) {
        Write-Success "✓ Services started"
    } else {
        Write-Error "✗ Failed to start services"
        exit 1
    }
}

function Stop-Compose {
    Write-Warning "Stopping docker-compose..."
    docker-compose -f docker/docker-compose.yml down
    Write-Success "✓ Services stopped"
}

# Main execution
switch ($Command.ToLower()) {
    "build" {
        Build-Image
    }
    "run" {
        Stop-Container
        Start-Container
    }
    "run-dev" {
        Stop-Container
        Start-DevContainer
    }
    "test" {
        Test-Container
    }
    "shell" {
        Open-Shell
    }
    "stop" {
        Stop-Container
    }
    "logs" {
        Show-Logs
    }
    "clean" {
        Clean-All
    }
    "compose-up" {
        Start-Compose
    }
    "compose-down" {
        Stop-Compose
    }
    default {
        Show-Help
    }
}