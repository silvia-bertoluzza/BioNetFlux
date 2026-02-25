# BioNetFlux Copilot Contract

You are assisting with a scientific HDG research code.

1. Read AGENT_RULES_SHORT.md and AGENT_INSTRUCTIONS.md before doing anything.
2. The mathematical formulation (weak form, static condensation, Newton, constraints) is fixed and owned by me. Never change it without approval.
3. Before coding: (a) restate task, (b) list files to modify/create, (c) flag any mathematical or architectural decisions.
4. If solver behavior, indexing, or data flow is affected, stop and ask.
5. Prefer clarity over cleverness. No metaprogramming.
6. Do not refactor or reorganize unless explicitly requested.
7. Only approved libraries: numpy, matplotlib, sympy, toml/tomllib/tomli, pytest, pandas.
8. No duplication, no dead code.
9. All new functionality requires pytest tests.


Confirm you understand and will follow this contract before proceeding.


_______________

HDG research code. Math formulation is fixed and mine.
Never change weak form, static condensation, Newton, constraints, or indexing without approval.
Only numpy, matplotlib, sympy, tomllib/tomli, pytest allowed.
Before coding: restate task, list files, flag math/architecture decisions.
If solver behavior or data flow is affected, stop and ask.
Prefer explicit, readable code. No refactors unless requested.
All new code requires pytest tests.
Confirm compliance before proceeding.

________________

For debugging:

HDG research code. Mathematical formulation is fixed and mine.
Do NOT change weak forms, static condensation, Newton logic, constraints, or indexing.

Goal is bug fixing only.

Before coding: restate the bug, list files to touch, explain suspected cause.
If fix affects solver behavior or data flow, stop and ask.

Only numpy, matplotlib, sympy, tomllib/tomli, pytest allowed.
Prefer minimal, localized changes. No refactors unless requested.
Add a regression pytest for every bug.

Confirm understanding before proceeding.

_________________

For adding new PDEs

HDG research code. Mathematical formulation is owned by me.

Before any implementation:
1. Restate the PDE system and unknown ordering.
2. List files/modules to create or modify.
3. Describe static condensation blocks and constraints.

Do NOT implement weak forms or condensation without approval.

Only numpy, matplotlib, sympy, tomllib/tomli, pytest allowed.
No architectural changes or new patterns.
All new problems require config, tests, and example scripts.

Confirm and wait before coding.


________________


For refactoring

HDG research code. Mathematics and algorithms are fixed.

This session is performance/refactoring only.
Do NOT change weak forms, condensation, Newton strategy, or constraints.

Before coding: state performance goal, list files, explain approach.
No new libraries without approval.

Prefer transparent optimizations (loops, array reuse, caching).
No clever abstractions.
All refactors must preserve behavior and keep code readable.

Confirm understanding before proceeding.