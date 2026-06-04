#!/usr/bin/env python3
"""Tests for the nonlinear-solver selection helper in
``examples/arc_example_ooc.py``.

The example exposes a ``--solver newton|picard`` switch that maps to
``build_optional_picard_solver``. Returning ``None`` keeps the TimeStepper's
default Newton solver; returning a ``PicardSolver`` makes the TimeStepper use
Picard (see ``time_stepper.py``). These tests cover that mapping without
running the full plotting/file-writing example.
"""
import importlib.util
import os

import pytest

from bionetflux.time_integration.picard_solver import PicardSolver

# ---------------------------------------------------------------------------
# Load the example module by file path (examples/ is not a package).
# ---------------------------------------------------------------------------
_EXAMPLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "examples", "arc_example_ooc.py"
)
_spec = importlib.util.spec_from_file_location("arc_example_ooc", _EXAMPLE_PATH)
arc_example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arc_example)


@pytest.mark.unit
class TestBuildOptionalPicardSolver:
    """The solver-selection helper used by the --solver CLI flag."""

    def test_newton_returns_none(self):
        """'newton' yields None, so TimeStepper uses its default Newton solver."""
        assert arc_example.build_optional_picard_solver("newton") is None

    def test_picard_returns_configured_solver(self):
        """'picard' yields a PicardSolver matching the upwind example settings."""
        solver = arc_example.build_optional_picard_solver("picard")
        assert isinstance(solver, PicardSolver)
        assert solver.tolerance == pytest.approx(1e-7)
        assert solver.max_iterations == 50

    def test_unknown_solver_raises(self):
        with pytest.raises(ValueError):
            arc_example.build_optional_picard_solver("bogus")
