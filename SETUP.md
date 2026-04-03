# BioNetFlux Development Environment Setup Guide

This guide helps you set up a consistent development environment across different machines and operating systems.

## Quick Start

### 1. Clone and Navigate
```bash
git clone https://github.com/silvia-bertoluzza/bionetflux.git
cd bionetflux
```

### 2. Choose Your Environment Setup

#### Option A: Editable install (Recommended)
```bash
# Create environment
python -m venv bionetflux-env
source bionetflux-env/bin/activate  # Windows: bionetflux-env\Scripts\activate

# Install in editable mode
pip install -e ".[dev]"
```

#### Option B: Conda + editable install
```bash
conda create -n bionetflux python=3.11
conda activate bionetflux
pip install -e ".[dev]"
```

#### Option C: Minimal Installation
```bash
pip install -e .
pip install pytest
```

### 3. Configure VS Code

1. **Copy settings template**:
   ```bash
   cp .vscode/settings.json.template .vscode/settings.json
   ```

2. **Set Python interpreter** in VS Code:
   - Open Command Palette: `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type "Python: Select Interpreter"
   - Choose your virtual environment Python

3. **Install recommended extensions**:
   - Python (`ms-python.python`)
   - Pylance (`ms-python.vscode-pylance`)
   - Python Test Explorer (`littlefoxteam.vscode-python-test-adapter`)

### 4. Verify Installation
```bash
# Run test suite
pytest

# Run an example
python examples/evolution_example_ks.py
```

## Platform-Specific Notes

### Windows
- Use PowerShell or Command Prompt
- Virtual environment activation: `bionetflux-env\Scripts\activate`
- Use forward slashes in Python paths

### macOS
- May need Xcode command line tools: `xcode-select --install`
- Use Terminal or iTerm2
- Virtual environment activation: `source bionetflux-env/bin/activate`

### Linux
- Install Python development headers: `sudo apt-get install python3-dev` (Ubuntu/Debian)
- Virtual environment activation: `source bionetflux-env/bin/activate`

## Testing Your Setup

### 1. Basic Import Test
```python
from bionetflux.core.problem import Problem
from bionetflux.geometry.domain_geometry import DomainGeometry
print("✓ BioNetFlux imports successful")
```

### 2. Run Test Suite
```bash
pytest -v
# Should show all tests passing
```

### 3. VS Code Integration Test
- Open VS Code in project directory
- Look for "Test Explorer" in Explorer sidebar
- Tests should be automatically discovered
- Try running a test by clicking the play button

## Troubleshooting

### Common Issues

**ImportError: No module named 'bionetflux'**
```bash
# Solution: install the package in editable mode
pip install -e .
```

**Tests not discovered in VS Code**
1. Check Python interpreter: `Cmd+Shift+P` → "Python: Select Interpreter"
2. Reload window: `Cmd+Shift+P` → "Developer: Reload Window"
3. Check `.vscode/settings.json` exists and is configured

**Package installation failures**
1. Ensure virtual environment is activated
2. Upgrade pip: `pip install --upgrade pip`
3. Try minimal installation: `pip install -r requirements-minimal.txt`

**Platform-specific dependency issues**
1. Check if you have a C compiler (needed for some packages)
2. Consider using conda for scientific packages: `conda install numpy scipy matplotlib`
3. Refer to individual package documentation

### Getting Help
1. Check [Issues](https://github.com/silvia-bertoluzza/bionetflux/issues) for known problems
2. Review [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development setup
3. Create new issue if problem persists

## Environment Files Summary

- `pyproject.toml` - Package configuration and dependencies
- `requirements.txt` - Production dependencies (for pip install -r)
- `requirements-dev.txt` - Complete development environment
- `requirements-minimal.txt` - Bare minimum for testing
- `requirements-core.txt` - Core scientific computing dependencies

## Next Steps

Once your environment is set up:
1. Read the [main documentation](docs/BioNetFlux_Documentation.md)
2. Try the [examples](examples/)
3. Review the [contributing guide](CONTRIBUTING.md) if you plan to contribute
4. Explore the codebase starting with `src/bionetflux/`