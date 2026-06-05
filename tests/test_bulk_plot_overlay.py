#!/usr/bin/env python3
"""Tests for overlaying discontinuous-Galerkin bulk solutions on the
``plot_2d_curves`` output of ``LeanMatplotlibPlotter``.

The overlay draws one solid segment per element for each equation, plotted
independently so jumps between elements appear as genuine discontinuities
(no vertical connectors, no markers). Bulk data layout is (2*neq, n_elements):
for equation ``eq`` and element ``m``, row ``2*eq`` is the left-node value and
row ``2*eq+1`` is the right-node value, spanning ``[nodes[m], nodes[m+1]]``.
"""
import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")  # headless backend for tests
import matplotlib.pyplot as plt

from bionetflux.core.bulk_data import BulkData
from bionetflux.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter


class _MockProblem:
    def __init__(self, neq=2, name="dom", domain_start=0.0, domain_length=1.0):
        self.neq = neq
        self.name = name
        self.domain_start = domain_start
        self.domain_length = domain_length
        # extrema: 2D endpoints used only for geometry/bounding box
        self.extrema = [(domain_start, 0.0), (domain_start + domain_length, 0.0)]


class _MockDiscretization:
    def __init__(self, n_elements=4, domain_start=0.0, domain_length=1.0):
        self.n_elements = n_elements
        self.nodes = np.linspace(domain_start, domain_start + domain_length,
                                 n_elements + 1)
        self.element_sizes = np.full(n_elements, domain_length / n_elements)


@pytest.fixture
def setup_single_domain():
    """One domain, neq=2, 4 elements (5 nodes)."""
    neq, n_elements = 2, 4
    problem = _MockProblem(neq=neq)
    disc = _MockDiscretization(n_elements=n_elements)
    plotter = LeanMatplotlibPlotter(problems=[problem], discretizations=[disc])
    n_nodes = n_elements + 1
    # Trace: neq * n_nodes flattened (per-equation blocks)
    trace = np.arange(neq * n_nodes, dtype=float)
    # Bulk array (2*neq, n_elements) with distinct values per row/element
    bulk_arr = np.arange(2 * neq * n_elements, dtype=float).reshape(2 * neq, n_elements)
    return plotter, [trace], bulk_arr, disc, neq, n_elements


@pytest.mark.unit
def test_no_bulk_is_backward_compatible(setup_single_domain):
    """Omitting bulk_solutions leaves the original behavior unchanged."""
    plotter, traces, _, _, _, _ = setup_single_domain
    fig = plotter.plot_2d_curves(traces)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


@pytest.mark.unit
def test_bulk_adds_one_segment_per_element_per_equation(setup_single_domain):
    """Bulk overlay adds exactly neq*n_elements Line2D segments to the axis."""
    plotter, traces, bulk_arr, _, neq, n_elements = setup_single_domain

    fig_plain = plotter.plot_2d_curves(traces, show_bounding_box=False,
                                       show_mesh_points=False)
    n_plain = len(fig_plain.axes[0].lines)
    plt.close(fig_plain)

    fig_bulk = plotter.plot_2d_curves(traces, show_bounding_box=False,
                                      show_mesh_points=False,
                                      bulk_solutions=[bulk_arr])
    n_bulk = len(fig_bulk.axes[0].lines)
    plt.close(fig_bulk)

    assert n_bulk - n_plain == neq * n_elements


@pytest.mark.unit
def test_bulk_segments_are_independent_two_point_lines(setup_single_domain):
    """Each element is its own 2-point segment (no connectors across elements)."""
    plotter, traces, bulk_arr, disc, neq, n_elements = setup_single_domain
    fig = plotter.plot_2d_curves(traces, show_bounding_box=False,
                                 show_mesh_points=False,
                                 bulk_solutions=[bulk_arr])
    ax = fig.axes[0]
    # The trace curves span all nodes; bulk segments have exactly 2 points.
    two_point_lines = [ln for ln in ax.lines if len(ln.get_xdata()) == 2]
    assert len(two_point_lines) == neq * n_elements
    plt.close(fig)


@pytest.mark.unit
def test_bulk_segment_coordinates_match_data(setup_single_domain):
    """A specific element's segment uses the correct (x, value) endpoints."""
    plotter, traces, bulk_arr, disc, neq, n_elements = setup_single_domain
    fig = plotter.plot_2d_curves(traces, show_bounding_box=False,
                                 show_mesh_points=False,
                                 bulk_solutions=[bulk_arr])
    ax = fig.axes[0]

    # Expected segment for equation 0, element 0
    eq_idx, elem_idx = 0, 0
    x_left, x_right = disc.nodes[elem_idx], disc.nodes[elem_idx + 1]
    c_left = bulk_arr[2 * eq_idx, elem_idx]
    c_right = bulk_arr[2 * eq_idx + 1, elem_idx]

    found = False
    for ln in ax.lines:
        xd, yd = np.asarray(ln.get_xdata()), np.asarray(ln.get_ydata())
        if (len(xd) == 2
                and np.allclose(xd, [x_left, x_right])
                and np.allclose(yd, [c_left, c_right])):
            found = True
            break
    assert found, "Expected bulk segment for eq0/elem0 not found"
    plt.close(fig)


@pytest.mark.unit
def test_accepts_bulkdata_objects(setup_single_domain):
    """bulk_solutions accepts BulkData objects (via get_data), not just arrays."""
    plotter, traces, bulk_arr, disc, neq, n_elements = setup_single_domain
    problem = _MockProblem(neq=neq)
    bd = BulkData(problem, disc, dual=False)
    bd.set_data(bulk_arr.copy())

    fig = plotter.plot_2d_curves(traces, show_bounding_box=False,
                                 show_mesh_points=False,
                                 bulk_solutions=[bd])
    two_point_lines = [ln for ln in fig.axes[0].lines if len(ln.get_xdata()) == 2]
    assert len(two_point_lines) == neq * n_elements
    plt.close(fig)
