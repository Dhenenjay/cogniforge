# Makefile for CogniForge project
# Compatible with Windows (PowerShell/cmd) and Unix-like systems

# Variables
PYTHON := python
POETRY := poetry
PROJECT_NAME := cogniforge
SRC_DIR := cogniforge
TEST_DIR := tests

# Detect OS for platform-specific commands
ifeq ($(OS),Windows_NT)
    # Windows specific
    RM := powershell -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
    MKDIR := powershell -Command "New-Item -ItemType Directory -Force"
    ENV_ACT := .venv\Scripts\activate
    PYTHON_VENV := .venv\Scripts\python.exe
    SEP := \\
else
    # Unix-like (Linux, macOS, Git Bash)
    RM := rm -rf
    MKDIR := mkdir -p
    ENV_ACT := .venv/bin/activate
    PYTHON_VENV := .venv/bin/python
    SEP := /
endif

# Colors for output (works in most terminals)
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

# Phony targets (not files)
.PHONY: help install dev run-api fmt lint test clean check all pre-commit pre-commit-update pre-commit-install pre-commit-uninstall

# Help target - displays available commands
help: ## Show this help message
	@echo.
	@echo $(BLUE)CogniForge Makefile$(NC)
	@echo $(YELLOW)==================$(NC)
	@echo.
	@echo Available targets:
	@echo.
	@echo   $(GREEN)install$(NC)    - Install production dependencies using Poetry
	@echo   $(GREEN)dev$(NC)        - Install all dependencies including dev tools
	@echo   $(GREEN)run-api$(NC)    - Run the FastAPI application with auto-reload
	@echo   $(GREEN)fmt$(NC)        - Format code using black and ruff
	@echo   $(GREEN)lint$(NC)       - Run linting checks (ruff, mypy)
	@echo   $(GREEN)test$(NC)       - Run test suite with pytest
	@echo   $(GREEN)clean$(NC)      - Remove cache files and build artifacts
	@echo   $(GREEN)check$(NC)      - Run all checks (lint + test)
	@echo   $(GREEN)all$(NC)        - Run dev + check
	@echo.
	@echo Use 'make [target]' to run a specific target
	@echo.

# Install production dependencies
install: ## Install production dependencies
	@echo $(BLUE)Installing production dependencies...$(NC)
	@$(POETRY) install --only main
	@echo $(GREEN)✓ Production dependencies installed$(NC)

# Install all dependencies including dev tools
dev: ## Install all dependencies (including dev)
	@echo $(BLUE)Installing all dependencies...$(NC)
	@$(POETRY) install
	@echo $(GREEN)✓ All dependencies installed$(NC)
	@echo $(YELLOW)Tip: Activate the virtual environment with: poetry shell$(NC)

# Run the FastAPI application
run-api: ## Run the FastAPI application
	@echo $(BLUE)Starting FastAPI application...$(NC)
	@$(POETRY) run uvicorn $(PROJECT_NAME).main:app --reload --host 0.0.0.0 --port 8000

# Alternative run target
run: run-api

# Format code
fmt: ## Format code with black and isort
	@echo $(BLUE)Formatting code with black...$(NC)
	@$(POETRY) run black $(SRC_DIR) $(TEST_DIR) --line-length 100
	@echo $(GREEN)✓ Black formatting complete$(NC)
	@echo.
	@echo $(BLUE)Organizing imports with isort...$(NC)
	@$(POETRY) run isort $(SRC_DIR) $(TEST_DIR) --profile black --line-length 100
	@echo $(GREEN)✓ Import organization complete$(NC)
	@echo.
	@echo $(BLUE)Fixing with ruff...$(NC)
	@$(POETRY) run ruff check --fix $(SRC_DIR) $(TEST_DIR)
	@echo $(GREEN)✓ Ruff fixes applied$(NC)
	@echo.
	@echo $(GREEN)✓ Code formatting complete!$(NC)

# Alternative format target
format: fmt

# Run linting checks
lint: ## Run linting checks
	@echo $(BLUE)Running flake8 linter...$(NC)
	@$(POETRY) run flake8 $(SRC_DIR) $(TEST_DIR) --max-line-length 100
	@echo $(GREEN)✓ Flake8 check passed$(NC)
	@echo.
	@echo $(BLUE)Running ruff linter...$(NC)
	@$(POETRY) run ruff check $(SRC_DIR) $(TEST_DIR)
	@echo $(GREEN)✓ Ruff check passed$(NC)
	@echo.
	@echo $(BLUE)Checking import order with isort...$(NC)
	@$(POETRY) run isort --check-only --diff $(SRC_DIR) $(TEST_DIR)
	@echo $(GREEN)✓ Import order check passed$(NC)
	@echo.
	@echo $(BLUE)Running mypy type checker...$(NC)
	@$(POETRY) run mypy $(SRC_DIR)
	@echo $(GREEN)✓ Type checking passed$(NC)
	@echo.
	@echo $(GREEN)✓ All linting checks passed!$(NC)

# Run tests
test: ## Run test suite
	@echo $(BLUE)Running pytest...$(NC)
	@$(POETRY) run pytest $(TEST_DIR) -v --tb=short
	@echo $(GREEN)✓ All tests passed!$(NC)

# Run tests with coverage
test-cov: ## Run tests with coverage report
	@echo $(BLUE)Running pytest with coverage...$(NC)
	@$(POETRY) run pytest $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html
	@echo $(GREEN)✓ Coverage report generated in htmlcov/$(NC)

# Clean up cache and build files
clean: ## Remove cache files and build artifacts
	@echo $(BLUE)Cleaning up...$(NC)
ifeq ($(OS),Windows_NT)
	@$(RM) "__pycache__"
	@$(RM) ".pytest_cache"
	@$(RM) ".mypy_cache"
	@$(RM) ".ruff_cache"
	@$(RM) "dist"
	@$(RM) "build"
	@$(RM) "*.egg-info"
	@$(RM) ".coverage"
	@$(RM) "htmlcov"
	@$(RM) "*.pyc"
	@powershell -Command "Get-ChildItem -Path . -Include __pycache__ -Recurse -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
else
	@$(RM) __pycache__ .pytest_cache .mypy_cache .ruff_cache
	@$(RM) dist build *.egg-info
	@$(RM) .coverage htmlcov
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
endif
	@echo $(GREEN)✓ Cleanup complete$(NC)

# Run all checks (lint + test)
check: ## Run all checks (lint + test)
	@echo $(BLUE)Running all checks...$(NC)
	@echo.
	@$(MAKE) lint
	@echo.
	@$(MAKE) test
	@echo.
	@echo $(GREEN)✓ All checks passed successfully!$(NC)

# Full development setup and check
all: ## Install dev dependencies and run all checks
	@$(MAKE) dev
	@echo.
	@$(MAKE) check

# Initialize the project (first time setup)
init: ## Initialize project (install poetry, dependencies, and pre-commit)
	@echo $(BLUE)Initializing project...$(NC)
	@echo Checking for Poetry installation...
ifeq ($(OS),Windows_NT)
	@where poetry >nul 2>&1 || (echo $(RED)Poetry not found! Please install it first: pip install poetry$(NC) && exit 1)
else
	@which poetry >/dev/null 2>&1 || (echo "$(RED)Poetry not found! Please install it first: pip install poetry$(NC)" && exit 1)
endif
	@echo $(GREEN)✓ Poetry found$(NC)
	@$(MAKE) dev
	@echo.
	@echo $(BLUE)Setting up pre-commit hooks...$(NC)
	@$(POETRY) run pre-commit install --install-hooks
	@echo $(GREEN)✓ Pre-commit hooks installed$(NC)
	@echo.
	@echo $(GREEN)✓ Project initialized successfully!$(NC)

# Pre-commit commands
pre-commit: ## Run pre-commit hooks on all files
	@echo $(BLUE)Running pre-commit hooks on all files...$(NC)
	@$(POETRY) run pre-commit run --all-files
	@echo $(GREEN)✓ Pre-commit checks complete$(NC)

pre-commit-update: ## Update pre-commit hooks to latest versions
	@echo $(BLUE)Updating pre-commit hooks...$(NC)
	@$(POETRY) run pre-commit autoupdate
	@echo $(GREEN)✓ Pre-commit hooks updated$(NC)

pre-commit-install: ## Install pre-commit hooks
	@echo $(BLUE)Installing pre-commit hooks...$(NC)
	@$(POETRY) run pre-commit install --install-hooks
	@echo $(GREEN)✓ Pre-commit hooks installed$(NC)

pre-commit-uninstall: ## Uninstall pre-commit hooks
	@echo $(BLUE)Uninstalling pre-commit hooks...$(NC)
	@$(POETRY) run pre-commit uninstall
	@echo $(GREEN)✓ Pre-commit hooks uninstalled$(NC)

# Run the API in production mode (no auto-reload)
run-prod: ## Run API in production mode
	@echo $(BLUE)Starting FastAPI in production mode...$(NC)
	@$(POETRY) run uvicorn $(PROJECT_NAME).main:app --host 0.0.0.0 --port 8000

# Update dependencies
update: ## Update all dependencies to latest versions
	@echo $(BLUE)Updating dependencies...$(NC)
	@$(POETRY) update
	@echo $(GREEN)✓ Dependencies updated$(NC)

# Show current Python and Poetry info
info: ## Show Python and Poetry environment info
	@echo $(BLUE)Environment Information:$(NC)
	@echo.
	@echo Python version:
	@$(PYTHON) --version
	@echo.
	@echo Poetry version:
	@$(POETRY) --version
	@echo.
	@echo Virtual environment:
	@$(POETRY) env info
	@echo.
	@echo $(GREEN)✓ Environment info displayed$(NC)

# Quick command to run black, ruff, and tests
quick: ## Quick check: format and test
	@$(MAKE) fmt
	@echo.
	@$(MAKE) test
	@echo.
	@echo $(GREEN)✓ Quick check complete!$(NC)