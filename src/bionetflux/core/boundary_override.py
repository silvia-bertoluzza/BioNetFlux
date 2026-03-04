"""
Boundary condition override utility.

Applies per-point, per-equation boundary condition overrides specified in a
TOML configuration to an existing ConstraintManager.  The default constraints
(homogeneous Neumann at every exterior boundary) are created by
``setup_constraints_from_geometry``.  This module replaces selected defaults
using the ``find_constraints`` / ``replace_constraint`` API.

TOML format
-----------
::

    [boundary_conditions]
    # Key: "<point_name>_<equation_name>"
    # Value: inline table with at least "type"

    B1_u  = { type = "dirichlet", data = "zeros" }
    B2_phi = { type = "neumann",  data = "sin_t" }
    B3_v   = { type = "robin",    alpha = 1.0, beta = 0.5, data = "zeros" }
"""

from typing import Dict, Any, List, Optional, Callable

from .constraints import ConstraintManager, ConstraintType


def apply_boundary_overrides(
    constraint_manager: ConstraintManager,
    boundary_overrides: Dict[str, Any],
    boundary_point_map: Dict[str, tuple],
    equation_names: List[str],
    function_resolver: Optional[Any] = None,
) -> None:
    """Replace default Neumann BCs with overrides specified in TOML config.

    For every key in *boundary_overrides* the function:

    1. Parses the key as ``<point_name>_<equation_name>``.
    2. Looks up ``(domain_id, position)`` from *boundary_point_map*.
    3. Finds the existing (default Neumann) constraint via
       ``constraint_manager.find_constraints``.
    4. Creates the requested constraint (Dirichlet / Neumann / Robin).
    5. Replaces the old constraint in-place.

    The caller **must** call ``constraint_manager.map_to_discretizations``
    after this function returns.

    Args:
        constraint_manager: The ConstraintManager populated by
            ``setup_constraints_from_geometry``.
        boundary_overrides: Dict from TOML ``[boundary_conditions]``.
            Keys are ``"<point>_<equation>"``; values are dicts with at
            least ``"type"`` and optionally ``"data"``, ``"alpha"``,
            ``"beta"``.
        boundary_point_map: ``{point_name: (domain_id, parameter)}`` stored
            in geometry global metadata by ``create_maze_geometry``.
        equation_names: Ordered list of equation names (e.g.
            ``['u', 'omega', 'v', 'phi']``).
        function_resolver: A ``FunctionResolver`` instance used to convert
            the ``"data"`` string into a callable ``f(s, t)``.  If *None*,
            data functions are not resolved (string names are kept as-is;
            useful only for testing).

    Raises:
        ValueError: On unknown point names, equation names, BC types, or
            when the expected constraint cannot be found.
    """

    if not boundary_overrides:
        return  # nothing to do

    for key, spec in boundary_overrides.items():
        # ------------------------------------------------------------------
        # 1. Parse key  →  point_name, equation_name
        # ------------------------------------------------------------------
        # The equation name is always the last token after '_'.
        # The point name is everything before that last '_'.
        parts = key.rsplit("_", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid boundary_conditions key '{key}'. "
                f"Expected format '<point_name>_<equation_name>'."
            )
        point_name, eq_name = parts

        # ------------------------------------------------------------------
        # 2. Resolve point → (domain_id, position)
        # ------------------------------------------------------------------
        if point_name not in boundary_point_map:
            available = ", ".join(sorted(boundary_point_map.keys()))
            raise ValueError(
                f"Unknown boundary point '{point_name}' in key '{key}'. "
                f"Available boundary points: {available}"
            )
        domain_id, position = boundary_point_map[point_name]

        # ------------------------------------------------------------------
        # 3. Resolve equation name → equation_index
        # ------------------------------------------------------------------
        if eq_name not in equation_names:
            raise ValueError(
                f"Unknown equation '{eq_name}' in key '{key}'. "
                f"Available equations: {equation_names}"
            )
        eq_idx = equation_names.index(eq_name)

        # ------------------------------------------------------------------
        # 4. Find the existing default Neumann constraint
        # ------------------------------------------------------------------
        indices = constraint_manager.find_constraints(
            domain_index=domain_id,
            equation_index=eq_idx,
            constraint_type=ConstraintType.NEUMANN,
            position=position,
        )
        if len(indices) != 1:
            raise ValueError(
                f"Expected exactly 1 Neumann constraint for point "
                f"'{point_name}' (domain {domain_id}, eq '{eq_name}') at "
                f"position {position}, found {len(indices)}."
            )
        old_idx = indices[0]

        # ------------------------------------------------------------------
        # 5. Parse the spec dict
        # ------------------------------------------------------------------
        if not isinstance(spec, dict):
            raise ValueError(
                f"Value for boundary_conditions key '{key}' must be a "
                f"table/dict, got {type(spec).__name__}."
            )

        bc_type = spec.get("type")
        if bc_type is None:
            raise ValueError(
                f"Missing 'type' in boundary_conditions entry '{key}'."
            )
        bc_type = bc_type.lower()

        # Resolve the optional data function
        data_func: Optional[Callable] = None
        data_name = spec.get("data")
        if data_name is not None and function_resolver is not None:
            data_func = function_resolver.resolve_function(data_name)

        # ------------------------------------------------------------------
        # 6. Create the replacement constraint
        # ------------------------------------------------------------------
        if bc_type == "dirichlet":
            new_constraint = constraint_manager.make_dirichlet(
                eq_idx, domain_id, position,
                data_function=data_func,
            )
        elif bc_type == "neumann":
            new_constraint = constraint_manager.make_neumann(
                eq_idx, domain_id, position,
                data_function=data_func,
            )
        elif bc_type == "robin":
            alpha = spec.get("alpha")
            beta = spec.get("beta")
            if alpha is None or beta is None:
                raise ValueError(
                    f"Robin BC at '{key}' requires both 'alpha' and 'beta'."
                )
            new_constraint = constraint_manager.make_robin(
                eq_idx, domain_id, position,
                alpha=float(alpha),
                beta=float(beta),
                data_function=data_func,
            )
        else:
            raise ValueError(
                f"Unknown BC type '{bc_type}' in key '{key}'. "
                f"Supported types: dirichlet, neumann, robin."
            )

        # ------------------------------------------------------------------
        # 7. Replace
        # ------------------------------------------------------------------
        constraint_manager.replace_constraint(old_idx, new_constraint)
        print(f"  ✓ BC override: {point_name} eq '{eq_name}' → {bc_type}"
              + (f" (data={data_name})" if data_name else ""))
