# Contributing to BioNetFlux

Thank you for your interest in contributing to BioNetFlux! This guide will help you set up your development environment and understand our testing practices.

## Development Environment Setup

### 1. Fork and Clone
```bash
git fork https://github.com/silvia-bertoluzza/bionetflux
git clone https://github.com/YOUR_USERNAME/bionetflux.git
cd bionetflux
```

### 2. Create Development Environment

#### Option A: Using venv
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

#### Option B: Using Conda
```bash
conda create -n bionetflux-dev python=3.11
conda activate bionetflux-dev
pip install -e ".[dev]"
```

The editable install (`-e`) sets up the package so that code changes take effect immediately without reinstalling.

## Testing

### Test Structure
```
tests/
├── conftest.py             # Shared test fixtures
├── test_sample.py          # Basic functionality tests
├── test_geometry.py        # Geometry module tests
├── test_problem.py         # Problem definition tests
├── test_bulk_data.py       # Data management tests
└── ...                     # Additional test modules
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_geometry.py

# Specific test function
pytest tests/test_geometry.py::TestBasicFunctionality::test_empty_geometry_creation

# With coverage report
pytest --cov=bionetflux --cov-report=html

# Verbose output
pytest -v
```

### Test Markers
- `@pytest.mark.unit`: Fast, isolated unit tests
- `@pytest.mark.integration`: Tests involving multiple components
- `@pytest.mark.slow`: Long-running tests

### Writing Tests

1. **File Naming**: `test_*.py` for test files
2. **Function Naming**: `test_*` for test functions
3. **Class Naming**: `Test*` for test classes
4. Use `np.allclose()` for numerical comparisons with appropriate tolerances
5. Use Google-style docstrings

Example:

```python
"""Tests for geometry functionality."""

import pytest
import numpy as np
from bionetflux.geometry.domain_geometry import DomainGeometry


class TestGeometryBasics:
    """Test basic geometry functionality."""

    def test_empty_geometry_creation(self):
        """Test creating empty geometry."""
        geometry = DomainGeometry("test_network")
        assert geometry.name == "test_network"
        assert len(geometry.domains) == 0

    @pytest.mark.slow
    def test_large_geometry_performance(self):
        """Test performance with large geometries."""
        pass
```

## Code Quality

### Style Guidelines
- Follow PEP 8
- Use type hints on all function signatures
- Google-style docstrings
- Prefer mathematical clarity over Python idioms

### Running Quality Checks
```bash
flake8 src/ tests/
mypy src/
black src/ tests/
```

## Submitting Changes

1. Create feature branch: `git checkout -b feature/new-feature`
2. Write tests for new functionality
3. Ensure all tests pass: `pytest`
4. Update documentation if needed
5. Submit pull request with clear description

### Commit Messages
```
core/constraints: add Robin boundary condition support

- Implement Robin BC in ConstraintManager
- Add comprehensive test suite
- Update documentation with examples
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.