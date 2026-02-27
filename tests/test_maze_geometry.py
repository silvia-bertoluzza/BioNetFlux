"""
Pytest tests for the create_maze_geometry function.

Tests cover:
- Correct number of domains and connections
- Point classification (B, M, G)
- Axis-alignment validation
- Interior and boundary connection counts
- Parameter values at connections
- Geometry validation passes
- Error handling for bad inputs
"""

import os
import tempfile
import pytest
import numpy as np

from bionetflux.geometry.domain_geometry import (
    create_maze_geometry,
    DomainGeometry,
    EXTERIOR_BOUNDARY,
)


# ---------------------------------------------------------------------------
#  Path to the shipped maze_1_data CSV files
# ---------------------------------------------------------------------------
_MAZE_1_DIR = os.path.join(
    os.path.dirname(__file__), "..", "src", "bionetflux", "geometry", "maze_1_data"
)


class TestMazeGeometryCreation:
    """Test that create_maze_geometry produces the expected geometry."""

    @pytest.fixture(scope="class")
    def maze(self) -> DomainGeometry:
        """Create the maze geometry once for the whole test class."""
        return create_maze_geometry(_MAZE_1_DIR)

    # --- Domain counts ---

    def test_number_of_domains(self, maze: DomainGeometry):
        """There should be 25 segments (lines) in the maze."""
        assert maze.num_domains() == 25

    def test_domain_names_match_line_ids(self, maze: DomainGeometry):
        """Each domain name should be L1 … L25."""
        names = sorted(maze.get_domain_names(), key=lambda n: int(n[1:]))
        expected = [f"L{i}" for i in range(1, 26)]
        assert names == expected

    # --- Connection counts ---

    def test_total_connections(self, maze: DomainGeometry):
        """Total connections should be 36 (13 boundary + 23 interior)."""
        assert maze.num_connections() == 36

    def test_boundary_connections(self, maze: DomainGeometry):
        """13 boundary connections expected (13 G-points, each in 1 segment)."""
        assert len(maze.get_boundary_connections()) == 13

    def test_interior_connections(self, maze: DomainGeometry):
        """23 interior connections: 14 from B-points + 9 from M-points."""
        assert len(maze.get_interior_connections()) == 23

    # --- Axis alignment ---

    def test_all_segments_axis_aligned(self, maze: DomainGeometry):
        """Every segment must be either horizontal or vertical."""
        for domain in maze:
            dx = abs(domain.extrema_end[0] - domain.extrema_start[0])
            dy = abs(domain.extrema_end[1] - domain.extrema_start[1])
            is_horizontal = dy < 1e-12
            is_vertical = dx < 1e-12
            assert is_horizontal or is_vertical, (
                f"Domain {domain.name} is diagonal: "
                f"{domain.extrema_start} → {domain.extrema_end}"
            )

    # --- Geometry validation ---

    def test_validate_geometry(self, maze: DomainGeometry):
        """Geometry validation must report no errors."""
        assert maze.validate_geometry(verbose=False)

    # --- Positive segment lengths ---

    def test_positive_lengths(self, maze: DomainGeometry):
        """Every segment must have positive Euclidean length."""
        for domain in maze:
            assert domain.euclidean_length() > 0, (
                f"Domain {domain.name} has zero length"
            )

    # --- Boundary connections are exterior ---

    def test_boundary_connections_are_exterior(self, maze: DomainGeometry):
        """All boundary connections should be EXTERIOR_BOUNDARY type."""
        for conn in maze.get_boundary_connections():
            assert conn.domain2_id == EXTERIOR_BOUNDARY

    # --- Interior connections have valid parameters ---

    def test_interior_connection_parameters_in_range(self, maze: DomainGeometry):
        """Parameter values of interior connections must lie within domain ranges."""
        for conn in maze.get_interior_connections():
            dom1 = maze.get_domain(conn.domain1_id)
            dom2 = maze.get_domain(conn.domain2_id)
            assert dom1.domain_start - 1e-9 <= conn.parameter1 <= dom1.domain_start + dom1.domain_length + 1e-9, (
                f"parameter1={conn.parameter1} out of range for domain {conn.domain1_id}"
            )
            assert dom2.domain_start - 1e-9 <= conn.parameter2 <= dom2.domain_start + dom2.domain_length + 1e-9, (
                f"parameter2={conn.parameter2} out of range for domain {conn.domain2_id}"
            )

    # --- Spot-check a few specific segments ---

    def test_segment_L1_extrema(self, maze: DomainGeometry):
        """L1 goes from B0(0,0) to G0(4.5, 0) — horizontal."""
        dom = maze.get_domain(maze.find_domain_by_name("L1"))
        assert dom.extrema_start == (0.0, 0.0)
        assert dom.extrema_end == (4.5, 0.0)
        assert abs(dom.domain_length - 4.5) < 1e-12

    def test_segment_L11_extrema(self, maze: DomainGeometry):
        """L11 goes from B12(0,6) to B13(6,6) — horizontal."""
        dom = maze.get_domain(maze.find_domain_by_name("L11"))
        assert dom.extrema_start == (0.0, 6.0)
        assert dom.extrema_end == (6.0, 6.0)
        assert abs(dom.domain_length - 6.0) < 1e-12

    def test_segment_L15_vertical(self, maze: DomainGeometry):
        """L15 is B4(1,1.5) to B9(1,3.5) — vertical at x=1."""
        dom = maze.get_domain(maze.find_domain_by_name("L15"))
        assert abs(dom.extrema_start[0] - 1.0) < 1e-12
        assert abs(dom.extrema_end[0] - 1.0) < 1e-12
        assert abs(dom.domain_length - 2.0) < 1e-12


class TestMazeGeometryDefaultDir:
    """Test that the default data_dir resolves correctly."""

    def test_default_dir_works(self):
        """create_maze_geometry() without arguments should find maze_1_data."""
        geom = create_maze_geometry()
        assert geom.num_domains() == 25


class TestMazeGeometryErrors:
    """Test error handling for bad CSV data."""

    def _write_csv(self, path: str, header: str, rows: list):
        """Helper: write a simple CSV file."""
        with open(path, "w") as fh:
            fh.write(header + "\n")
            for row in rows:
                fh.write(",".join(str(v) for v in row) + "\n")

    def test_missing_file_raises(self, tmp_path):
        """FileNotFoundError when CSV files are missing."""
        with pytest.raises(FileNotFoundError):
            create_maze_geometry(str(tmp_path))

    def test_unknown_point_type_raises(self, tmp_path):
        """ValueError on unrecognised point-type prefix."""
        self._write_csv(str(tmp_path / "points.csv"), "ID,x,y",
                        [["X0", 0, 0], ["X1", 1, 0]])
        self._write_csv(str(tmp_path / "lines.csv"), "LineID,StartPointID,EndPointID",
                        [["L1", "X0", "X1"]])
        with pytest.raises(ValueError, match="Unknown point type"):
            create_maze_geometry(str(tmp_path))

    def test_diagonal_segment_raises(self, tmp_path):
        """ValueError on a diagonal segment."""
        self._write_csv(str(tmp_path / "points.csv"), "ID,x,y",
                        [["G0", 0, 0], ["G1", 1, 1]])
        self._write_csv(str(tmp_path / "lines.csv"), "LineID,StartPointID,EndPointID",
                        [["L1", "G0", "G1"]])
        with pytest.raises(ValueError, match="diagonal"):
            create_maze_geometry(str(tmp_path))

    def test_undefined_point_in_lines_raises(self, tmp_path):
        """ValueError when lines.csv references a point not in points.csv."""
        self._write_csv(str(tmp_path / "points.csv"), "ID,x,y",
                        [["G0", 0, 0]])
        self._write_csv(str(tmp_path / "lines.csv"), "LineID,StartPointID,EndPointID",
                        [["L1", "G0", "G1"]])
        with pytest.raises(ValueError, match="not in points.csv"):
            create_maze_geometry(str(tmp_path))

    def test_b_point_with_one_segment_raises(self, tmp_path):
        """ValueError when a B-point is extremum of < 2 segments."""
        self._write_csv(str(tmp_path / "points.csv"), "ID,x,y",
                        [["B0", 0, 0], ["G0", 1, 0]])
        self._write_csv(str(tmp_path / "lines.csv"), "LineID,StartPointID,EndPointID",
                        [["L1", "B0", "G0"]])
        with pytest.raises(ValueError, match="fewer than 2"):
            create_maze_geometry(str(tmp_path))

    def test_m_point_no_through_segment_raises(self, tmp_path):
        """ValueError when an M-point has no through-segment."""
        # M0 is the extremum of L1 but no segment passes through (3,0).
        self._write_csv(str(tmp_path / "points.csv"), "ID,x,y",
                        [["M0", 3, 0], ["G0", 0, 0], ["G1", 5, 0]])
        self._write_csv(str(tmp_path / "lines.csv"), "LineID,StartPointID,EndPointID",
                        [["L1", "G0", "M0"], ["L2", "M0", "G1"]])
        with pytest.raises(ValueError, match="no through-segment"):
            create_maze_geometry(str(tmp_path))


class TestMinimalMaze:
    """Test on a small hand-crafted maze to verify connection logic exactly."""

    @pytest.fixture
    def tiny_maze_dir(self, tmp_path):
        """Create a minimal T-junction maze:

        G0 ----B0---- G1
                |
               M0  (T-junction on the horizontal segment)
                |
               G2

        Segments:
          L1: G0(0,0) → B0(2,0)   horizontal
          L2: B0(2,0) → G1(4,0)   horizontal
          L3: M0(2,1) → G2(2,3)   vertical, with M0 interior to no segment
          ... wait, M0 should be interior to one segment. Let me redesign:

        G0 ----M0---- G1       (horizontal: (0,0)→(4,0), M0 at (2,0))
                |
               G2               (vertical: M0(2,0)→G2(2,2))

        But M0 is an endpoint of the vertical and interior to the horizontal.
        The horizontal is one long segment: L1 from G0(0,0) to G1(4,0).
        The vertical is: L2 from M0(2,0) to G2(2,2).
        """
        points = [["G0", 0, 0], ["G1", 4, 0], ["M0", 2, 0], ["G2", 2, 2]]
        lines_data = [["L1", "G0", "G1"], ["L2", "M0", "G2"]]

        points_path = str(tmp_path / "points.csv")
        lines_path = str(tmp_path / "lines.csv")

        with open(points_path, "w") as fh:
            fh.write("ID,x,y\n")
            for row in points:
                fh.write(",".join(str(v) for v in row) + "\n")

        with open(lines_path, "w") as fh:
            fh.write("LineID,StartPointID,EndPointID\n")
            for row in lines_data:
                fh.write(",".join(row) + "\n")

        return str(tmp_path)

    def test_tiny_maze_domains(self, tiny_maze_dir):
        geom = create_maze_geometry(tiny_maze_dir)
        assert geom.num_domains() == 2

    def test_tiny_maze_boundary_count(self, tiny_maze_dir):
        """3 G-points → 3 boundary connections."""
        geom = create_maze_geometry(tiny_maze_dir)
        assert len(geom.get_boundary_connections()) == 3

    def test_tiny_maze_interior_count(self, tiny_maze_dir):
        """1 M-point → 1 interior connection (L2 extremum ↔ L1 interior)."""
        geom = create_maze_geometry(tiny_maze_dir)
        assert len(geom.get_interior_connections()) == 1

    def test_tiny_maze_t_junction_parameter(self, tiny_maze_dir):
        """The T-junction connects L2 at param 0 to L1 at param 2.0."""
        geom = create_maze_geometry(tiny_maze_dir)
        interior = geom.get_interior_connections()
        assert len(interior) == 1
        conn = interior[0]
        # L2 (domain 1) connects at its start (param=0) to L1 (domain 0)
        # at the interior point (2,0), which is at param=2.0 on L1
        # (L1 runs from x=0 to x=4, length=4, so param at x=2 is 2.0)
        dom_l2 = geom.find_domain_by_name("L2")
        dom_l1 = geom.find_domain_by_name("L1")

        if conn.domain1_id == dom_l2:
            assert abs(conn.parameter1 - 0.0) < 1e-12  # start of L2
            assert abs(conn.parameter2 - 2.0) < 1e-12  # interior of L1
            assert conn.domain2_id == dom_l1
        else:
            # connection stored in reverse order
            assert conn.domain1_id == dom_l1
            assert abs(conn.parameter1 - 2.0) < 1e-12
            assert conn.domain2_id == dom_l2
            assert abs(conn.parameter2 - 0.0) < 1e-12

    def test_tiny_maze_validates(self, tiny_maze_dir):
        geom = create_maze_geometry(tiny_maze_dir)
        assert geom.validate_geometry(verbose=False)
