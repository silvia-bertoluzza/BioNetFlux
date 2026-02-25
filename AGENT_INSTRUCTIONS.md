# Agent Instructions for BioNetFlux Development

> **Purpose**: This document tells AI coding agents how to work on the BioNetFlux codebase. Follow these instructions strictly. When in doubt, **ask before acting**.

---

## 1. About the Developer

- I have **advanced mathematical expertise** (numerical methods, PDEs, functional analysis, HDG methods) but **limited hands-on programming experience**.
- I understand code design basics but I am not a software engineer.
- I rely on agents for implementation, but **I own the mathematical algorithm**. Every choice that affects the mathematical formulation, discretization, or solver behavior must be discussed with me before implementation.
- Explain programming concepts when relevant, but never simplify or hand-wave the mathematics.

---

## 2. About the Project

BioNetFlux is a **scientific computing research code** for solving systems of PDEs on 1D networks of segments using the **Hybridizable Discontinuous Galerkin (HDG)** method.

### Key characteristics

| Aspect | Detail |
|--------|--------|
| Language | Python 3.11+ |
| Method | HDG with static condensation, implicit Euler time integration, Newton-Raphson nonlinear solver |
| Domain | 1D network of segments managed by `DomainGeometry` |
| Equations | Parametric number of coupled equations per problem (2 for Keller-Segel, 4 for Organ-on-Chip, extensible) |
| Heritage | Ported from a MATLAB reference implementation |
| Purpose | **Methodological research** — the code itself is part of the research output |

### Architecture overview

```
User script (examples/)
  → SolverSetup (setup_solver.py)
    → Problem module (problems/) — defines geometry, physics, ICs, BCs
    → DomainGeometry (geometry/) — network topology
    → GlobalDiscretization (core/) — mesh + time stepping parameters
    → ConstraintManager (core/) — BCs and junction conditions
  → Time loop
    → TimeStepper → NewtonSolver → GlobalAssembler
      → domain_flux_jump() per domain (static condensation)
      → ConstraintManager residuals
    → BulkDataManager updates
  → Post-processing: error evaluation, visualization
```

---

## 3. Golden Rules

### 3.1 Mathematical algorithm is mine

- **Never** change the mathematical formulation (weak form, quadrature rules, basis functions, static condensation procedure, Newton iteration structure, constraint enforcement) without explicit approval.
- **Never** substitute a numerical algorithm with a library call (e.g., replacing the Newton solver with `scipy.optimize`, or using a mesh library for discretization) without asking first.
- When proposing algorithmic changes, explain the mathematical implications clearly.
- **Never** refactor for style, structure, or “cleanliness” unless explicitly asked.

### 3.2 External libraries require approval

Approved libraries (use freely):
- `numpy` — array operations, linear algebra (`numpy.linalg.solve`, `numpy.linalg.cond`)
- `matplotlib` — visualization
- `sympy` — symbolic integration for elementary matrices and function parsing
- `toml` / `tomllib` / `tomli` — TOML configuration parsing
- `pytest` — testing
- `pandas` - linear regression

**Any other library** must be proposed and justified before use. This includes `scipy` submodules, `numba`, `jax`, `fenics`, `meshio`, or any other scientific computing package. I need to understand and control the methodology—black-box solvers are unacceptable.

### 3.3 Simplicity over cleverness

- Write code that a **mathematician with basic Python knowledge** can read and understand.
- Prefer explicit loops with clear variable names over dense one-liners.
- Use straightforward control flow. Avoid metaprogramming, decorators beyond `@property`/`@dataclass`/`@abstractmethod`/`@classmethod`, or advanced Python magic unless there is a clear justification.
- When there is a choice between "Pythonic" idiom and mathematical clarity, **choose mathematical clarity**.

### 3.4 No duplication

- **Never** duplicate functions. If similar logic exists, refactor to share it.
- **Never** duplicate data. Each piece of information should have a single authoritative source.
- Before writing new code, search the codebase for existing functionality that does the same or a similar thing.
- If you find duplication in existing code, flag it and propose a consolidation.

### 3.5 No dead code

- Do not leave commented-out code, unused imports, or unreachable branches.
- When refactoring, remove the old code after the new code is verified.

### 3.6 Things that always need my approval

- Changes to the weak formulation or variational form
- Changes to the static condensation procedure
- Changes to the Newton solver convergence criteria or update strategy
- Addition of new external library dependencies
- Changes to the data flow between major components (Problem → Discretization → Solver)
- Removal or renaming of public APIs
- Architectural changes (new design patterns, reorganization of modules)

---

## 4. Coding Standards

### 4.1 Naming conventions

| Entity | Convention | Example |
|--------|-----------|---------|
| Classes | PascalCase | `BulkDataManager`, `GlobalAssembler` |
| Functions / methods | snake_case | `compute_forcing_terms`, `domain_flux_jump` |
| Variables | snake_case | `trace_solutions`, `n_elements` |
| Constants | UPPER_SNAKE_CASE | `EXTERIOR_BOUNDARY`, `PERIODIC_BOUNDARY` |
| Private members | single underscore prefix | `_ensure_initialized`, `_implementations` |
| Files | snake_case | `lean_bulk_data_manager.py` |

Mathematical variable names (`neq`, `dt`, `tau`, `chi`, `dchi`, `mu`, `nu`) are acceptable and encouraged when they match the mathematical notation.

### 4.2 Type hints

- All function signatures must have type annotations for parameters and return values.
- Use `typing` module types: `List`, `Optional`, `Tuple`, `Dict`, `Callable`, `Union`.
- Use `np.ndarray` for NumPy array type hints.

### 4.3 Docstrings

Use Google-style docstrings:

```python
def compute_flux(self, solution: np.ndarray, dt: float) -> np.ndarray:
    """Compute the numerical flux from trace solution values.

    Evaluates the HDG numerical flux using the stabilization parameter tau
    and the current trace solution.

    Args:
        solution: Trace solution vector of shape (n_trace_dofs,).
        dt: Time step size.

    Returns:
        Flux vector of shape (n_trace_dofs,).
    """
```

- Every public class and method must have a docstring.
- Include mathematical context in docstrings when relevant (reference to equations, MATLAB equivalences).

### 4.4 Data structures

- Use `@dataclass` for value objects and result containers (e.g., `NewtonResult`, `TimeStepResult`, `DomainInfo`).
- Use NumPy arrays as the primary numerical data structure.
- Prefer explicit named fields over generic dicts/tuples for structured data.

### 4.5 Design patterns in use

| Pattern | Where | Purpose |
|---------|-------|---------|
| Abstract Base Class | `StaticCondensationBase`, `BaseConfigManager` | Define contracts for extensible components |
| Factory | `StaticCondensationFactory` | Map problem types to implementations |
| Factory method | `GlobalAssembler.from_framework_objects()` | Convenient construction from higher-level objects |
| Lazy initialization | `SolverSetup` properties | Defer expensive construction until needed |

Do not introduce new design patterns without discussion. Keep the architecture flat and transparent.

---

## 5. Project Structure

```
BioNetFlux/
├── config/                    # TOML parameter files
├── examples/                  # User-facing scripts
├── src/
│   ├── setup_solver.py        # Orchestrator (SolverSetup, quick_setup)
│   └── bionetflux/
│       ├── core/              # Problem, Discretization, Constraints,
│       │                      #   BulkData, FluxJump, GlobalAssembly,
│       │                      #   StaticCondensation, ErrorEvaluator
│       ├── geometry/          # DomainGeometry, DomainInfo
│       ├── problems/          # Problem definitions (KS, OoC, templates)
│       ├── solver/            # (placeholder — solver logic in time_integration/)
│       ├── time_integration/  # NewtonSolver, TimeStepper, TimeStepResult
│       ├── utils/             # ElementaryMatrices, ConfigManager
│       ├── visualization/     # Plotters (Lean, MultiDomain, Solution)
│       └── analysis/          # ErrorEvaluator (convergence studies)
├── tests/                     # pytest test files
├── docs/                      # LaTeX and Markdown documentation
└── validation_scripts/        # Validation and comparison scripts
```

### Where things go

- **New PDE problem types** → `src/bionetflux/problems/` (new module with `create_global_framework()`)
- **New static condensation** → `src/bionetflux/core/` (subclass `StaticCondensationBase`, register in factory)
- **New constraint types** → `src/bionetflux/core/constraints.py`
- **Numerical utilities** → `src/bionetflux/utils/`
- **Plotting features** → `src/bionetflux/visualization/`
- **Error analysis** → `src/bionetflux/analysis/`
- **Configuration schemas** → `src/bionetflux/problems/` (new config manager subclass)

---

## 6. How to Add a New Problem Type

This is the most common extension task. The steps are:

1. **Create a problem module** in `src/bionetflux/problems/` (e.g., `new_problem.py`).
2. **Implement `create_global_framework(geometry, config_file)`** that returns `(problems, global_discretization, constraint_manager, problem_name)`.
3. **Implement the static condensation** for the new equation system in `src/bionetflux/core/` by subclassing `StaticCondensationBase`.
4. **Register** the new SC class in `StaticCondensationFactory`.
5. **Create a config manager** in `src/bionetflux/problems/` by subclassing `BaseConfigManager`.
6. **Add a TOML config file** in `config/`.
7. **Add tests** in `tests/`.
8. **Add an example script** in `examples/`.

Ask me to validate the mathematical formulation (weak form, static condensation blocks) at step 3 before proceeding with implementation.

---

## 7. Working with the Geometry Module

`DomainGeometry` (`src/bionetflux/geometry/domain_geometry.py`) is the central data structure for network topology.

- Domains are 1D segments with directed orientation and (x, y) extrema.
- Connections can be interior junctions (trace continuity, Kedem-Katchalsky) or exterior boundaries (Dirichlet, Neumann).
- Builder functions (`build_arc_sequence_geometry`, `build_grid_geometry`) provide convenient construction.
- **Do not bypass the geometry module**. All domain connectivity must go through `DomainGeometry`.
- When modifying geometry code, ensure all existing tests pass and network validation remains intact.

---

## 8. Testing Requirements

- **All new code must have tests.** Place tests in `tests/test_<module>.py`.
- Use `pytest` with class-based test organization (`class TestFeatureName`).
- Use `@pytest.mark.unit` for unit tests, `@pytest.mark.integration` for integration tests, `@pytest.mark.slow` for expensive tests.
- Use `np.allclose()` for numerical comparisons with appropriate tolerances.
- **Run existing tests before and after any change** to verify nothing breaks.
- When fixing a bug, add a regression test that would have caught it.

---

## 9. Configuration (TOML) System

- Parameters are stored in TOML files under `config/`.
- Each problem type has a config manager (subclass of `BaseConfigManager`) that defines defaults.
- Function-valued parameters (initial conditions, forcing, exact solutions) are specified as string names resolved by `FunctionResolver`.
- **Do not hardcode** physical parameters in source code. They belong in TOML config or as `Problem` attributes.

---

## 10. Communication Protocol

### Before starting work

1. **State what you understand** the task to be.
2. **List the files** you plan to modify or create.
3. **Identify any mathematical or algorithmic decisions** that need my input.
4. Wait for confirmation before proceeding with implementation if the task involves changes to the algorithm, the architecture, or the addition of external dependencies.

### During implementation

- Make **small, incremental changes** that can be reviewed individually.
- After each logical step, briefly state what was done and what comes next.
- **Run tests** after changes and report results.
- If you encounter existing code that seems wrong or inconsistent, **ask before changing it** — it may be intentional.

### When proposing changes

- For **bug fixes**: describe the bug, its cause, and the fix.
- For **refactoring**: explain what duplication or complexity is being removed and why.
- For **new features**: describe the interface (function signatures, class API) before implementing the body.
- For **algorithmic changes**: explain the mathematical impact in terms I can evaluate.


### Things you can do autonomously

- Bug fixes that don't change the algorithm (off-by-one errors, wrong variable names, missing imports)
- Adding or improving docstrings
- Adding type hints
- Formatting and style fixes
- Adding tests for existing functionality
- Improving error messages
- Improving visualization (new plot types, better formatting)
- Refactoring for clarity without changing behavior (extracting helper functions, renaming private variables)

---

## 11. Common Pitfalls

- **Trace vs bulk indexing**: Trace unknowns live at element interfaces; bulk unknowns are interior to elements. These have different sizes and orderings. Be very careful with indexing.
- **Multi-equation ordering**: With `neq` equations, arrays are often organized equation-by-equation. Verify indexing conventions before modifying array operations.
- **MATLAB heritage**: Some comments reference MATLAB function names (`scBlocks.m`, `StaticC.m`, `fluxJump.m`). These are useful for cross-referencing the reference implementation.
- **Constraint ordering**: The constraint system uses Lagrange multipliers. The ordering of constraints in the global system matters. Do not reorder without understanding the implications.
- **`TimeStepResult` duplication**: This dataclass currently exists in both `time_step_result.py` and `time_stepper.py`. This is a known issue to be resolved.

---

## 12. Version Control Notes

- The project uses Git with feature branches.
- Commit messages should be descriptive and reference the component being modified (e.g., "core/constraints: add Robin boundary condition support").
- Do not commit generated files (outputs, `__pycache__`, `.DS_Store`).
- Run tests before committing.

---

## 13. Summary of Key Files

| File | Role | Modify with care? |
|------|------|-------------------|
| `src/bionetflux/core/problem.py` | PDE problem definition | Yes — core data structure |
| `src/bionetflux/core/lean_global_assembly.py` | Global system assembly | Yes — affects solver correctness |
| `src/bionetflux/core/flux_jump.py` | Per-element flux balance | Yes — core numerical kernel |
| `src/bionetflux/core/static_condensation_*.py` | Element-level linear algebra | Yes — mathematical core |
| `src/bionetflux/core/constraints.py` | Boundary/junction conditions | Yes — affects well-posedness |
| `src/bionetflux/geometry/domain_geometry.py` | Network topology | Yes — foundational data structure |
| `src/bionetflux/utils/elementary_matrices.py` | Reference element matrices | Yes — mathematical foundation |
| `src/bionetflux/time_integration/newton_solver.py` | Nonlinear solver | Yes — convergence behavior |
| `src/bionetflux/time_integration/time_stepper.py` | Time integration | Yes — accuracy and stability |
| `src/setup_solver.py` | Orchestrator | Moderate — affects setup flow |
| `src/bionetflux/problems/*.py` | Problem definitions | Moderate — problem-specific |
| `src/bionetflux/visualization/*.py` | Plotting | Low risk |
| `src/bionetflux/utils/config_manager.py` | Configuration | Low risk |
