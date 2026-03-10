#!/usr/bin/env python3
"""
Pytest-compatible test script for the Problem class.

This script tests all functionality of the problem module including:
- Problem creation and initialization
- Parameter management
- Function setting and validation
- Problem validation
- Error handling
- Integration with different problem types

Usage:
    pytest test_problem.py
    pytest test_problem.py -v  # verbose output
    pytest test_problem.py -s  # show print statements
"""

import sys
import os
import numpy as np
import time
import pytest
from typing import Dict, Any

# sys.path hack — commented out, use pip install -e . instead
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bionetflux.core.problem import Problem


class TestBasicFunctionality:
    """Test basic problem operations."""

    def test_default_problem_creation(self):
        """Test default problem creation."""
        default_problem = Problem()
        assert default_problem.neq == 2
        assert default_problem.domain_start == 0.0
        assert default_problem.domain_length == 1.0
        assert default_problem.domain_end == 1.0
        assert default_problem.name == "unnamed_problem"
        assert default_problem.type == "keller_segel"

    def test_custom_problem_creation(self):
        """Test custom problem creation."""
        custom_problem = Problem(
            neq=3,
            domain_start=1.0,
            domain_length=2.0,
            parameters=np.array([1.0, 0.5, 2.0, 0.0]),
            problem_type="organ_on_chip",
            name="custom_problem"
        )
        
        assert custom_problem.neq == 3
        assert custom_problem.domain_start == 1.0
        assert custom_problem.domain_length == 2.0
        assert custom_problem.domain_end == 3.0
        assert custom_problem.name == "custom_problem"
        assert custom_problem.type == "organ_on_chip"


class TestParameterManagement:
    """Test parameter management functions."""

    def test_parameter_setting_and_getting(self):
        """Test parameter setting and getting."""
        problem = Problem(parameters=np.array([1.0, 2.0, 3.0]))
        
        # Test getting parameters
        assert problem.get_parameter(0) == 1.0
        assert problem.get_parameter(1) == 2.0
        assert problem.get_parameter(2) == 3.0
        
        # Test setting parameters
        problem.set_parameter(0, 1.5)
        problem.set_parameter(1, 2.5)
        problem.set_parameter(2, 3.5)
        
        assert problem.get_parameter(0) == 1.5
        assert problem.get_parameter(1) == 2.5
        assert problem.get_parameter(2) == 3.5

    def test_parameter_array_setting(self):
        """Test parameter array setting."""
        problem = Problem(parameters=np.array([1.0, 2.0]))
        problem.set_parameters(np.array([3.0, 4.0, 5.0]))
        
        assert problem.get_parameter(0) == 3.0
        assert problem.get_parameter(1) == 4.0
        assert problem.get_parameter(2) == 5.0


class TestFunctionSetting:
    """Test function setting and management."""

    def test_initial_condition_setting(self):
        """Test initial condition setting."""
        problem = Problem(neq=3)
        
        # Set initial conditions
        problem.set_initial_condition(0, lambda s: np.sin(s))
        problem.set_initial_condition(1, lambda s: np.cos(s))
        problem.set_initial_condition(2, lambda s: np.exp(-s))
        
        # Test functions
        test_s = np.array([0.0, 1.0, 2.0])
        result0 = problem.u0[0](test_s)
        result1 = problem.u0[1](test_s)
        result2 = problem.u0[2](test_s)
        
        assert np.allclose(result0, np.sin(test_s))
        assert np.allclose(result1, np.cos(test_s))
        assert np.allclose(result2, np.exp(-test_s))

    def test_force_function_setting(self):
        """Test force function setting."""
        problem = Problem(neq=2)
        
        # Set force functions
        problem.set_force(0, lambda s, t: s + t)
        problem.set_force(1, lambda s, t: s * t)
        
        # Test evaluation
        test_s = np.array([1.0, 2.0, 3.0])
        test_t = 2.0
        result0 = problem.force[0](test_s, test_t)
        result1 = problem.force[1](test_s, test_t)
        
        assert np.allclose(result0, test_s + test_t)
        assert np.allclose(result1, test_s * test_t)

    def test_solution_function_setting(self):
        """Test solution function setting."""
        problem = Problem(neq=2)
        
        # Set solution functions
        problem.set_solution(0, lambda s, t: np.sin(s) * np.exp(-t))
        problem.set_solution(1, lambda s, t: np.cos(s) * np.exp(-t))
        
        # Test evaluation
        test_s = np.array([0.0, np.pi/2, np.pi])
        test_t = 1.0
        result0 = problem.solution[0](test_s, test_t)
        result1 = problem.solution[1](test_s, test_t)
        
        assert np.allclose(result0, np.sin(test_s) * np.exp(-test_t))
        assert np.allclose(result1, np.cos(test_s) * np.exp(-test_t))


class TestProblemValidation:
    """Test problem validation functionality."""

    def test_valid_problem_validation(self):
        """Test valid problem validation."""
        valid_problem = Problem(
            neq=2,
            domain_start=0.0,
            domain_length=1.0,
            parameters=np.array([1.0, 2.0, 0.1, 0.0]),
            problem_type="keller_segel",
            name="valid_test"
        )
        
        is_valid = valid_problem.validate_problem(verbose=False)
        assert is_valid == True

    def test_invalid_problem_negative_domain(self):
        """Test invalid problem with negative domain length."""
        invalid_problem = Problem(
            neq=2,
            domain_start=0.0,
            domain_length=-1.0,
            parameters=np.array([1.0, 2.0, 0.1, 0.0]),
            problem_type="keller_segel",
            name="invalid_test"
        )
        
        is_valid = invalid_problem.validate_problem(verbose=False)
        assert is_valid == False


class TestProblemTypes:
    """Test different problem types."""

    def test_keller_segel_problem(self):
        """Test Keller-Segel problem."""
        ks_problem = Problem(
            neq=2,
            parameters=np.array([1.0, 2.0, 0.1, 0.0]),
            problem_type="keller_segel",
            name="KS_test"
        )
        
        # Set Keller-Segel specific functions
        ks_problem.set_chemotaxis(
            lambda x: np.ones_like(x),
            lambda x: np.zeros_like(x)
        )
        
        ks_problem.set_initial_condition(0, lambda s: np.exp(-s**2))
        ks_problem.set_initial_condition(1, lambda s: np.sin(np.pi * s))
        
        is_valid = ks_problem.validate_problem(verbose=False)
        functions_ok = ks_problem.test_functions(verbose=False)
        
        assert is_valid and functions_ok

    def test_organ_on_chip_problem(self):
        """Test Organ-on-Chip problem."""
        ooc_problem = Problem(
            neq=4,
            parameters=np.array([1e-9, 0.001, 1e-4, 1e-5, 1e-3]),
            problem_type="organ_on_chip",
            name="OoC_test"
        )
        
        # Set initial conditions
        ooc_problem.set_initial_condition(0, lambda s: 0.5 * np.ones_like(s))
        ooc_problem.set_initial_condition(1, lambda s: np.zeros_like(s))
        ooc_problem.set_initial_condition(2, lambda s: 0.1 * np.ones_like(s))
        ooc_problem.set_initial_condition(3, lambda s: 0.05 * np.ones_like(s))
        
        # Set source terms
        for i in range(4):
            ooc_problem.set_force(i, lambda s, t: np.zeros_like(s))
        
        is_valid = ooc_problem.validate_problem(verbose=False)
        functions_ok = ooc_problem.test_functions(verbose=False)
        
        assert is_valid and functions_ok


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_parameter_access(self):
        """Test invalid parameter access."""
        problem = Problem(parameters=np.array([1.0, 2.0]))
        
        # Should raise IndexError
        with pytest.raises(IndexError):
            problem.get_parameter(5)

    def test_invalid_function_setting(self):
        """Test invalid function setting."""
        problem = Problem()
        
        # Invalid function (not callable)
        with pytest.raises(TypeError):
            problem.set_function("invalid_func", 12345)


class TestPredefinedProblems:
    """Test the predefined test problems."""

    def test_create_test_problems(self):
        """Test creation of all predefined test problems."""
        test_problems = Problem.create_test_problems()
        
        expected_problems = ["ks_basic", "ooc_basic", "custom_geometry", "analytical", "invalid"]
        
        for name in expected_problems:
            assert name in test_problems, f"Missing test problem: {name}"

    @pytest.mark.parametrize("problem_name", ["ks_basic", "ooc_basic", "custom_geometry", "analytical"])
    def test_valid_predefined_problems(self, problem_name):
        """Test that valid predefined problems pass self-tests."""
        test_problems = Problem.create_test_problems()
        problem = test_problems[problem_name]
        
        test_result = problem.run_self_test(verbose=False)
        assert test_result, f"{problem_name} problem self-test failed"

    def test_invalid_predefined_problem(self):
        """Test that invalid predefined problem fails validation."""
        test_problems = Problem.create_test_problems()
        invalid_problem = test_problems["invalid"]
        
        is_valid = invalid_problem.validate_problem(verbose=False)
        assert not is_valid, "Invalid problem should fail validation"


class TestPerformance:
    """Test performance with function evaluations."""

    @pytest.mark.slow
    def test_function_evaluation_performance(self):
        """Test function evaluation performance."""
        problem = Problem(neq=2)
        
        # Set computationally simple functions
        problem.set_initial_condition(0, lambda s: np.sin(s))
        problem.set_initial_condition(1, lambda s: np.cos(s))
        problem.set_force(0, lambda s, t: s * t)
        problem.set_force(1, lambda s, t: s**2 + t**2)
        
        # Time function evaluations
        large_s = np.linspace(0, 10, 10000)
        t_val = 1.0
        
        start_time = time.time()
        
        for _ in range(100):  # Multiple evaluations
            result0 = problem.u0[0](large_s)
            result1 = problem.u0[1](large_s)
            result2 = problem.force[0](large_s, t_val)
            result3 = problem.force[1](large_s, t_val)
        
        eval_time = time.time() - start_time
        
        # Performance assertion
        assert eval_time < 5.0, f"Function evaluations too slow: {eval_time:.3f}s"


# Fixtures for reusable test objects
@pytest.fixture
def basic_problem():
    """Fixture providing a basic problem instance."""
    return Problem(
        neq=2,
        domain_start=0.0,
        domain_length=1.0,
        parameters=np.array([1.0, 2.0, 0.1, 0.0]),
        problem_type="keller_segel",
        name="test_problem"
    )


@pytest.fixture
def configured_problem(basic_problem):
    """Fixture providing a fully configured problem."""
    basic_problem.set_initial_condition(0, lambda s: np.exp(-s**2))
    basic_problem.set_initial_condition(1, lambda s: np.sin(np.pi * s))
    basic_problem.set_force(0, lambda s, t: np.zeros_like(s))
    basic_problem.set_force(1, lambda s, t: np.zeros_like(s))
    basic_problem.set_chemotaxis(
        lambda x: np.ones_like(x),
        lambda x: np.zeros_like(x)
    )
    return basic_problem


class TestFixtures:
    """Test using fixtures."""

    def test_basic_problem_fixture(self, basic_problem):
        """Test the basic problem fixture."""
        assert basic_problem.neq == 2
        assert basic_problem.type == "keller_segel"
        assert basic_problem.name == "test_problem"

    def test_configured_problem_fixture(self, configured_problem):
        """Test the configured problem fixture."""
        assert configured_problem.u0[0] is not None
        assert configured_problem.u0[1] is not None
        assert configured_problem.force[0] is not None
        assert configured_problem.force[1] is not None
        
        # Test function execution
        test_s = np.array([0.0, 0.5, 1.0])
        result = configured_problem.u0[0](test_s)
        assert isinstance(result, np.ndarray)


# Configure pytest markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


def main():
    """Run tests using pytest."""
    print("This file is now pytest-compatible!")
    print("Usage:")
    print("  pytest test_problem.py")
    print("  pytest test_problem.py -v")
    print("  pytest test_problem.py -s  # show prints")
    print("  pytest test_problem.py -m \"not slow\"  # skip slow tests")
    print("  pytest test_problem.py::TestBasicFunctionality::test_default_problem_creation")
    
    # Run with pytest for backwards compatibility
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    main()