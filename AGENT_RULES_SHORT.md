# BioNetFlux Agent Rules (Short)

1. Mathematical formulation belongs to the developer. Never change weak forms,
   static condensation, Newton structure, constraints, or discretization without approval.

4. Always:
   - State understanding of task
   - List files to modify
   - Flag mathematical decisions
   before coding.

2. Ask before:
   - algorithm changes
   - solver behavior changes
   - architecture changes
   - external dependencies.

3. Prioritize mathematical clarity over Python cleverness.
   Prefer explicit loops and readable variables.
   
2. Only approved libraries: numpy, matplotlib, sympy, toml/tomllib/tomli, pandas, pytest.
   Any others require explicit permission.

5. Use type hints and Google-style docstrings.

6. No duplicated logic, no dead code.

7. All new code requires pytest tests.


Always read AGENT_INSTRUCTIONS.md for full details.