# Pre-commit Setup and Usage Guide

## 📋 Overview

This project uses pre-commit hooks to automatically check and format code before commits. All tools are configured with a **line length of 100 characters**.

## 🛠 Tools Included

### Code Formatters
- **Black** - Uncompromising Python code formatter (line length: 100)
- **isort** - Sorts and organizes imports (line length: 100)
- **Prettier** - Formats YAML, JSON, and Markdown files

### Linters
- **Flake8** - Style guide enforcement (line length: 100)
  - flake8-bugbear - Additional bug detection
  - flake8-comprehensions - Comprehension improvements
  - flake8-simplify - Code simplification suggestions
  - flake8-docstrings - Docstring checking (Google style)
  - pep8-naming - Naming convention checks
- **Ruff** - Fast Python linter
- **Bandit** - Security vulnerability scanner

### Type Checking
- **mypy** - Static type checker

### File Checks
- Trailing whitespace removal
- End-of-file fixer
- YAML/JSON syntax validation
- Large file prevention (>1MB)
- Merge conflict detection

## 🚀 Quick Start

### 1. Initial Setup

```bash
# Install all dependencies including pre-commit
make dev

# Install pre-commit hooks
make pre-commit-install
```

### 2. Manual Pre-commit Run

```bash
# Run all hooks on all files
make pre-commit

# Or using poetry directly
poetry run pre-commit run --all-files
```

### 3. Run Specific Hooks

```bash
# Run only black
poetry run pre-commit run black --all-files

# Run only flake8
poetry run pre-commit run flake8 --all-files

# Run only isort
poetry run pre-commit run isort --all-files
```

## 📝 Configuration

All tools are configured to use **100 character line length**:

### Configuration Files
- `.pre-commit-config.yaml` - Pre-commit hook configuration
- `.flake8` - Flake8 specific settings
- `pyproject.toml` - Black, isort, and ruff settings
- `setup.cfg` - Additional tool configurations
- `.editorconfig` - Editor settings for consistency

## 🔧 Makefile Commands

```bash
# Format code with black and isort
make fmt

# Run all linting checks
make lint

# Run pre-commit on all files
make pre-commit

# Update pre-commit hooks to latest versions
make pre-commit-update

# Install pre-commit hooks
make pre-commit-install

# Uninstall pre-commit hooks
make pre-commit-uninstall
```

## 💡 Common Scenarios

### Fixing Import Order Issues

```bash
# Auto-fix import order
poetry run isort cogniforge tests

# Or use make
make fmt
```

### Fixing Code Formatting

```bash
# Auto-format with black
poetry run black cogniforge tests --line-length 100

# Or use make
make fmt
```

### Checking Without Fixing

```bash
# Check formatting without changes
poetry run black --check cogniforge tests
poetry run isort --check-only cogniforge tests
poetry run flake8 cogniforge tests
```

## 🔍 Pre-commit Workflow

1. **On Every Commit**: Pre-commit automatically runs configured hooks
2. **If Issues Found**: 
   - Formatters will auto-fix issues
   - Linters will report issues to fix manually
3. **After Fixes**: Stage changes and commit again

## ⚙️ Customizing Rules

### Ignoring Specific Rules

In code:
```python
# flake8: noqa: E501  # Ignore line length for this line
long_line_that_exceeds_100_characters = "This is acceptable"

# type: ignore  # Ignore mypy for this line
untyped_variable = some_untyped_function()
```

### Skipping Hooks Temporarily

```bash
# Skip pre-commit hooks for one commit (use sparingly!)
git commit --no-verify -m "Emergency fix"
```

## 📊 Line Length Configuration

All tools are configured for 100 character line length:

- **Black**: `line-length = 100` in `pyproject.toml`
- **isort**: `line_length = 100` in `pyproject.toml`
- **Flake8**: `max-line-length = 100` in `.flake8`
- **Ruff**: `line-length = 100` in `pyproject.toml`

## 🐛 Troubleshooting

### Pre-commit not running
```bash
# Reinstall hooks
make pre-commit-install
```

### Import order conflicts with Black
```bash
# isort is configured with `profile = "black"` to prevent conflicts
# If issues persist, run:
make fmt
```

### Type checking errors
```bash
# Install type stubs
poetry add --group dev types-requests types-pyyaml
```

## 📚 Additional Resources

- [Black Documentation](https://black.readthedocs.io/)
- [isort Documentation](https://pycqa.github.io/isort/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Ruff Documentation](https://beta.ruff.rs/docs/)

## ✅ Best Practices

1. **Always run `make fmt` before committing** to ensure consistent formatting
2. **Fix linting issues** rather than disabling rules
3. **Keep line length under 100** for better readability
4. **Write docstrings** in Google style format
5. **Use type hints** for better code documentation and error detection
6. **Run `make check`** before pushing to ensure all checks pass

## 🎯 GitHub Actions Integration

Pre-commit hooks can be integrated with CI/CD:

```yaml
# .github/workflows/pre-commit.yml
name: pre-commit

on:
  pull_request:
  push:
    branches: [main]

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - uses: pre-commit/action@v3.0.0
```