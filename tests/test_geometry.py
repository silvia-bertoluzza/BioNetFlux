#!/usr/bin/env python3
"""
Pytest-compatible test script for the DomainGeometry module.

This script tests all functionality of the geometry module including:
- Basic domain operations
- Geometry validation
- Connectivity analysis
- Parameter space management
- Error handling
- Performance with large geometries

Usage:
    pytest test_geometry.py
    pytest test_geometry.py -v  # verbose output
    pytest test_geometry.py::test_basic_functionality  # run specific test
"""

import sys
import os
import numpy as np
import time
from typing import Dict, Any
import pytest

# sys.path hack — commented out, use pip install -e . instead
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bionetflux.geometry import DomainGeometry, DomainInfo


class TestBasicFunctionality:
    """Test basic geometry operations."""

    def test_empty_geometry_creation(self):
        """Test empty geometry creation."""
        empty_geom = DomainGeometry("empty_test")
        assert len(empty_geom) == 0
        assert empty_geom.num_domains() == 0
        assert empty_geom.name == "empty_test"

    def test_domain_addition(self):
        """Test domain addition."""
        geom = DomainGeometry("test_geom")
        domain_id = geom.add_domain((0.0, 0.0), (1.0, 0.0), name="test_domain")
        assert domain_id == 0
        assert len(geom) == 1
        assert geom.get_domain(0).name == "test_domain"

    def test_domain_retrieval_and_properties(self):
        """Test domain retrieval and properties."""
        geom = DomainGeometry("test_geom")
        geom.add_domain((0.0, 0.0), (1.0, 0.0), name="test_domain")
        
        domain = geom.get_domain(0)
        assert domain.extrema_start == (0.0, 0.0)
        assert domain.extrema_end == (1.0, 0.0)
        assert abs(domain.euclidean_length() - 1.0) < 1e-12
        assert domain.center_point() == (0.5, 0.0)
        assert domain.direction_vector() == (1.0, 0.0)

    def test_multiple_domain_operations(self):
        """Test multiple domain operations."""
        geom = DomainGeometry("test_geom")
        geom.add_domain((0.0, 0.0), (1.0, 0.0), name="test_domain")
        geom.add_domain((1.0, 0.0), (1.0, 1.0), name="domain2")
        geom.add_domain((1.0, 1.0), (0.0, 1.0), name="domain3")
        
        assert len(geom) == 3
        assert len(geom.get_domain_names()) == 3
        assert geom.find_domain_by_name("domain2") == 1
        assert geom.find_domain_by_name("nonexistent") is None

    def test_bounding_box_calculation(self):
        """Test bounding box calculation."""
        geom = DomainGeometry("test_geom")
        geom.add_domain((0.0, 0.0), (1.0, 0.0), name="test_domain")
        geom.add_domain((1.0, 0.0), (1.0, 1.0), name="domain2")
        geom.add_domain((1.0, 1.0), (0.0, 1.0), name="domain3")
        
        bbox = geom.get_bounding_box()
        expected = {'x_min': 0.0, 'x_max': 1.0, 'y_min': 0.0, 'y_max': 1.0}
        for key, expected_val in expected.items():
            assert abs(bbox[key] - expected_val) < 1e-12, f"Bounding box {key} mismatch"


class TestValidationFunctionality:
    """Test geometry validation functionality."""

    def test_valid_geometry_validation(self):
        """Test valid geometry validation."""
        valid_geom = DomainGeometry("valid_test")
        valid_geom.add_domain((0.0, 0.0), (1.0, 0.0), name="seg1")
        valid_geom.add_domain((1.0, 0.0), (2.0, 0.0), name="seg2")
        
        is_valid = valid_geom.validate_geometry(verbose=False)
        assert is_valid == True

    def test_zero_length_domain_detection(self):
        """Test invalid geometry detection - zero length domains."""
        invalid_geom = DomainGeometry("invalid_test")
        invalid_geom.add_domain((0.0, 0.0), (0.0, 0.0), name="zero_length")
        
        is_valid = invalid_geom.validate_geometry(verbose=False)
        assert is_valid == False

    def test_duplicate_name_detection(self):
        """Test duplicate name detection."""
        dup_geom = DomainGeometry("duplicate_test")
        dup_geom.add_domain((0.0, 0.0), (1.0, 0.0), name="duplicate")
        dup_geom.add_domain((2.0, 0.0), (3.0, 0.0), name="duplicate")
        
        is_valid = dup_geom.validate_geometry(verbose=False)
        assert is_valid == False

    def test_overlapping_parameter_spaces_warning(self):
        """Test overlapping parameter spaces produce warnings, not errors."""
        overlap_geom = DomainGeometry("overlap_test")
        overlap_geom.add_domain((0.0, 0.0), (1.0, 0.0), 
                               domain_start=0.0, domain_length=1.5, name="domain1")
        overlap_geom.add_domain((1.0, 0.0), (2.0, 0.0), 
                               domain_start=1.0, domain_length=1.5, name="domain2")
        
        is_valid = overlap_geom.validate_geometry(verbose=False)
        # Should still be valid (overlapping parameters are just a warning)
        assert is_valid == True


class TestConnectivityAnalysis:
    """Test connectivity analysis functionality."""

    def test_connected_geometry_analysis(self):
        """Test connected geometry analysis."""
        connected_geom = DomainGeometry("connected_test")
        connected_geom.add_domain((0.0, 0.0), (1.0, 0.0), name="seg1")
        connected_geom.add_domain((1.0, 0.0), (2.0, 0.0), name="seg2")
        connected_geom.add_domain((2.0, 0.0), (2.0, 1.0), name="seg3")
        
        connectivity = connected_geom.get_connectivity_info()
        assert connectivity['is_connected'] == True
        assert connectivity['num_components'] == 1
        assert len(connectivity['intersections']) >= 2

    def test_disconnected_geometry_analysis(self):
        """Test disconnected geometry analysis."""
        disconnected_geom = DomainGeometry("disconnected_test")
        disconnected_geom.add_domain((0.0, 0.0), (1.0, 0.0), name="island1")
        disconnected_geom.add_domain((3.0, 3.0), (4.0, 3.0), name="island2")
        
        connectivity = disconnected_geom.get_connectivity_info()
        assert connectivity['is_connected'] == False
        assert connectivity['num_components'] == 2
        assert len(connectivity['isolated_domains']) == 2

    def test_intersection_detection(self):
        """Test intersection detection."""
        t_geom = DomainGeometry("t_junction_test")
        t_geom.add_domain((0.0, -1.0), (0.0, 1.0), name="vertical")
        t_geom.add_domain((-1.0, 0.0), (1.0, 0.0), name="horizontal")
        
        intersections = t_geom.find_intersections(tolerance=1e-6)
        # Should find intersection at (0, 0)
        assert len(intersections) >= 1


class TestParameterSpaceManagement:
    """Test parameter space management."""

    def test_parameter_space_suggestions(self):
        """Test parameter space suggestions."""
        geom = DomainGeometry("param_test")
        geom.add_domain((0.0, 0.0), (1.0, 0.0), domain_start=0.0, domain_length=1.0)
        geom.add_domain((1.0, 0.0), (3.0, 0.0), domain_start=0.5, domain_length=2.0)  # Overlapping
        
        suggestions = geom.suggest_parameter_spacing(gap=0.1)
        assert len(suggestions) == 2
        
        # Check that suggestions don't overlap
        start1, len1 = suggestions[0]
        start2, len2 = suggestions[1]
        assert start2 >= start1 + len1 + 0.1  # Should have gap

    def test_custom_parameter_spaces(self):
        """Test custom parameter spaces."""
        custom_geom = DomainGeometry("custom_param_test")
        custom_geom.add_domain((0.0, 0.0), (1.0, 0.0), 
                              domain_start=10.0, domain_length=5.0, name="custom1")
        custom_geom.add_domain((1.0, 0.0), (2.0, 0.0), 
                              domain_start=20.0, domain_length=3.0, name="custom2")
        
        domain1 = custom_geom.get_domain(0)
        domain2 = custom_geom.get_domain(1)
        
        assert domain1.domain_start == 10.0
        assert domain1.domain_length == 5.0
        assert domain2.domain_start == 20.0
        assert domain2.domain_length == 3.0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_domain_access(self):
        """Test invalid domain access raises appropriate errors."""
        geom = DomainGeometry("error_test")
        geom.add_domain((0.0, 0.0), (1.0, 0.0))
        
        # Should raise IndexError
        with pytest.raises(IndexError):
            geom.get_domain(5)

    def test_domain_removal(self):
        """Test domain removal functionality."""
        geom = DomainGeometry("removal_test")
        id1 = geom.add_domain((0.0, 0.0), (1.0, 0.0), name="remove_me")
        id2 = geom.add_domain((1.0, 0.0), (2.0, 0.0), name="keep_me")
        
        assert len(geom) == 2
        geom.remove_domain(id1)
        assert len(geom) == 1
        assert geom.get_domain(0).name == "keep_me"

    def test_empty_geometry_operations(self):
        """Test operations on empty geometry don't crash."""
        empty_geom = DomainGeometry("empty_ops_test")
        
        # These should not crash
        bbox = empty_geom.get_bounding_box()
        connectivity = empty_geom.get_connectivity_info()
        suggestions = empty_geom.suggest_parameter_spacing()
        
        assert len(suggestions) == 0
        assert connectivity['num_components'] == 0


class TestPredefinedGeometries:
    """Test the predefined test geometries."""

    def test_create_test_geometries(self):
        """Test creation of all predefined test geometries."""
        test_geometries = DomainGeometry.create_test_geometries()
        
        expected_geometries = ["linear", "t_junction", "grid", "star", "branching", "degenerate"]
        
        for name in expected_geometries:
            assert name in test_geometries, f"Missing test geometry: {name}"

    @pytest.mark.parametrize("geometry_name", ["linear", "t_junction", "grid", "star", "branching"])
    def test_valid_predefined_geometries(self, geometry_name):
        """Test that valid predefined geometries pass self-tests."""
        test_geometries = DomainGeometry.create_test_geometries()
        geom = test_geometries[geometry_name]
        
        # Run self-test on each geometry
        test_result = geom.run_self_test(verbose=False)
        assert test_result, f"{geometry_name} geometry self-test failed"

    def test_degenerate_geometry_fails_validation(self):
        """Test that degenerate geometry correctly fails validation."""
        test_geometries = DomainGeometry.create_test_geometries()
        degenerate_geom = test_geometries["degenerate"]
        
        # Degenerate case should fail validation
        is_valid = degenerate_geom.validate_geometry(verbose=False)
        assert not is_valid, "Degenerate geometry should be invalid"


class TestPerformance:
    """Test performance with larger geometries."""

    @pytest.mark.slow
    def test_large_geometry_creation(self):
        """Test creation of large geometries."""
        start_time = time.time()
        
        large_geom = DomainGeometry("large_test")
        n_domains = 1000
        
        for i in range(n_domains):
            large_geom.add_domain((i, 0), (i+1, 0), name=f"domain_{i}")
        
        creation_time = time.time() - start_time
        
        assert len(large_geom) == n_domains
        assert creation_time < 5.0, f"Creation time too slow: {creation_time:.3f}s"

    @pytest.mark.slow
    def test_large_geometry_operations(self):
        """Test operations on large geometries."""
        # Create large geometry
        large_geom = DomainGeometry("large_ops_test")
        n_domains = 100  # Smaller for faster testing
        
        for i in range(n_domains):
            large_geom.add_domain((i, 0), (i+1, 0), name=f"domain_{i}")
        
        start_time = time.time()
        
        # Test validation
        large_geom.validate_geometry(verbose=False)
        
        # Test bounding box
        bbox = large_geom.get_bounding_box()
        
        # Test connectivity
        connectivity = large_geom.get_connectivity_info()
        
        operation_time = time.time() - start_time
        assert operation_time < 10.0, f"Operations too slow: {operation_time:.3f}s"


# Fixtures for common test setups
@pytest.fixture
def simple_geometry():
    """Fixture providing a simple L-shaped geometry."""
    geom = DomainGeometry("simple_test")
    geom.add_domain((0.0, 0.0), (1.0, 0.0), name="horizontal")
    geom.add_domain((1.0, 0.0), (1.0, 1.0), name="vertical")
    return geom


@pytest.fixture
def complex_geometry():
    """Fixture providing a more complex test geometry."""
    geom = DomainGeometry("complex_test")
    geom.add_domain((0.0, 0.0), (1.0, 0.0), name="base")
    geom.add_domain((1.0, 0.0), (1.5, 0.5), name="branch1")
    geom.add_domain((1.0, 0.0), (1.5, -0.5), name="branch2")
    geom.add_domain((1.5, 0.5), (2.0, 0.5), name="continuation1")
    return geom


class TestFixtures:
    """Test using fixtures."""

    def test_simple_geometry_fixture(self, simple_geometry):
        """Test using the simple geometry fixture."""
        assert len(simple_geometry) == 2
        assert simple_geometry.validate_geometry(verbose=False)

    def test_complex_geometry_fixture(self, complex_geometry):
        """Test using the complex geometry fixture."""
        assert len(complex_geometry) == 4
        connectivity = complex_geometry.get_connectivity_info()
        assert connectivity['is_connected']


# Performance markers configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


def main():
    """Run all geometry tests using pytest."""
    print("This file is now pytest-compatible!")
    print("Run with: pytest test_geometry.py")
    print("Verbose: pytest test_geometry.py -v")
    print("Skip slow tests: pytest test_geometry.py -m \"not slow\"")
    print("Run specific test: pytest test_geometry.py::TestBasicFunctionality::test_empty_geometry_creation")
    
    # For backwards compatibility, can still run directly
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    main()
