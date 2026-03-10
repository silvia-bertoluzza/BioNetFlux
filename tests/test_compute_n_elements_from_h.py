"""Tests for compute_n_elements_from_h."""

import pytest
from bionetflux.core.discretization import compute_n_elements_from_h


class TestComputeNElementsFromH:
    """Unit tests for the h → n_elements helper."""

    def test_exact_division(self):
        """L / h is already an even integer."""
        assert compute_n_elements_from_h(10.0, 1.0) == 10

    def test_rounds_to_nearest_even(self):
        """L / h is odd → rounds to nearest even."""
        # L=9, h=1 → 9 → nearest even: round(9/2)=round(4.5)=4 → 2*4=8
        assert compute_n_elements_from_h(9.0, 1.0) == 8

    def test_minimum_is_four(self):
        """Very large h relative to L → clamps to 4."""
        assert compute_n_elements_from_h(1.0, 10.0) == 4

    def test_exact_four(self):
        """L / h = 4 exactly."""
        assert compute_n_elements_from_h(4.0, 1.0) == 4

    def test_result_is_even(self):
        """Result is always even for a variety of inputs."""
        for L in [3.7, 5.0, 12.3, 50.0, 100.0, 300.0]:
            for h in [0.5, 1.0, 2.5, 7.0, 20.0, 50.0]:
                n = compute_n_elements_from_h(L, h)
                assert n % 2 == 0, f"L={L}, h={h} → n={n} (odd!)"
                assert n >= 4, f"L={L}, h={h} → n={n} (< 4!)"

    def test_result_is_int(self):
        """Return type is int."""
        assert isinstance(compute_n_elements_from_h(10.0, 3.0), int)

    def test_typical_maze_domain(self):
        """Maze domain: L=50, h=20 → 50/20=2.5 → round(1.25)=1 → 2*1=2 → clamp to 4."""
        assert compute_n_elements_from_h(50.0, 20.0) == 4

    def test_long_domain_small_h(self):
        """L=300, h=20 → 300/20=15 → round(7.5)=8 → 2*8=16."""
        assert compute_n_elements_from_h(300.0, 20.0) == 16

    def test_short_domain(self):
        """L=10, h=20 → 10/20=0.5 → round(0.25)=0 → 2*0=0 → clamp to 4."""
        assert compute_n_elements_from_h(10.0, 20.0) == 4

    def test_rejects_zero_h(self):
        with pytest.raises(ValueError, match="positive"):
            compute_n_elements_from_h(10.0, 0.0)

    def test_rejects_negative_h(self):
        with pytest.raises(ValueError, match="positive"):
            compute_n_elements_from_h(10.0, -1.0)

    def test_rejects_zero_domain_length(self):
        with pytest.raises(ValueError, match="positive"):
            compute_n_elements_from_h(0.0, 1.0)

    def test_rejects_negative_domain_length(self):
        with pytest.raises(ValueError, match="positive"):
            compute_n_elements_from_h(-5.0, 1.0)
