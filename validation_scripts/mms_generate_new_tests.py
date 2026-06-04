#!/usr/bin/env python3
"""
Manufactured Solution (MMS) config generator for the OoC upwind problem.

For each test case the script:
  1. Computes forcing terms and Neumann boundary data symbolically.
  2. Writes a ready-to-run TOML configuration file in config/mms/.

PDE operators (1D, arc-length s, time t):
    du/dt  = nu  d2u/ds2  - d(chi*u*dphi/ds)/ds + f_u
    dw/dt  = eps d2w/ds2  - c*w + d*u            + f_omega
    dv/dt  = sig d2v/ds2  - lam*v                + f_v
    dphi/dt= mu  d2phi/ds2- a*phi + b*v           + f_phi

Neumann BC convention (n=-1 left, n=+1 right):
    B0_eq: data = -flux   (left)
    B1_eq: data = +flux   (right)

Usage:
    python validation_scripts/mms_ooc_upwind.py          # write all cases
    python validation_scripts/mms_ooc_upwind.py --list
    python validation_scripts/mms_ooc_upwind.py --case u_linear
    python validation_scripts/mms_ooc_upwind.py --dry-run --case u_sin
    python validation_scripts/mms_ooc_upwind.py --n-elements 80

Run a generated config:
    python examples/upwind_example_ooc.py \\
        --arc-number 1 --arc-length 6.28 \\
        config/mms/mms_<name>.toml
"""
import argparse
import os
import sys
import textwrap
from typing import Dict, Any

import sympy as sp

# ---------------------------------------------------------------------------
# Symbolic variables
# ---------------------------------------------------------------------------
s, t = sp.symbols('s t', real=True)
ZERO = sp.Integer(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simp(expr):
    try:
        return sp.simplify(expr)
    except Exception:
        return expr


def _str(expr) -> str:
    """Return 'zeros' for the zero expr, else sympy str (round-trips via sympify)."""
    if expr == ZERO or expr == 0:
        return "zeros"
    return str(expr)


def _chi_expr(chi_type: str, k1: float, k2: float, nu: float, phi_ms):
    """Symbolic chemotaxis sensitivity chi(phi) for the MMS forcing.

    Mirrors the solver definition in
    src/bionetflux/problems/ooc_problem_upwind.py:
        constant:            chi(phi) = k1
        receptor_saturation: chi(phi) = k1 / (nu * (k2 + phi)**2)
    The expression is evaluated at the manufactured chemoattractant phi.

    Args:
        chi_type: Chemotaxis model, "constant" or "receptor_saturation".
        k1: Cellular drift velocity coefficient.
        k2: Receptor dissociation constant.
        nu: Immune-cell diffusivity (rescaling factor in the solver).
        phi_ms: Manufactured chemoattractant expression in (s, t).

    Returns:
        The sympy chi(phi) expression, or None when k1 == 0 (chi vanishes),
        which reproduces the legacy chemotaxis-free forcing exactly.

    Raises:
        ValueError: If chi_type is not a supported chemotaxis model.
    """
    if k1 == 0:
        return None
    if chi_type == "constant":
        return sp.Float(k1) * sp.Float(nu)  # Rescale by diffusivity to match solver definition
    if chi_type == "receptor_saturation":
        return sp.Float(nu) * sp.Float(k1) / (sp.Float(k2) + phi_ms) ** 2
#        return sp.Float(k1) / (sp.Float(k2) + phi_ms) ** 2
    raise ValueError(f"Unsupported chemotaxis type: {chi_type}")


# ---------------------------------------------------------------------------
# MMS computation
# ---------------------------------------------------------------------------

def compute_mms(
    u_ms, omega_ms, v_ms, phi_ms,
    params: Dict[str, float],
    chi_ms=None,
    lam_ms=None,
) -> Dict[str, Any]:
    """
    Compute forcing terms and physical fluxes for manufactured solutions.

    chi_ms and lam_ms are sympy expressions in s, t evaluated at the
    manufactured solution. Pass None for the decoupled/linearised case.
    """
    nu_v  = params['nu']
    mu_v  = params['mu']
    eps_v = params['epsilon']
    sig_v = params['sigma']
    a_v   = params['a']
    b_v   = params['b']
    c_v   = params['c']
    d_v   = params['d']

    chi = chi_ms if chi_ms is not None else ZERO
    lam = lam_ms if lam_ms is not None else ZERO

    du   = sp.diff(u_ms,    s)
    domg = sp.diff(omega_ms, s)
    dv   = sp.diff(v_ms,    s)
    dphi = sp.diff(phi_ms,  s)

    flux_u     = nu_v * du   - chi * u_ms * dphi
    flux_omega = eps_v * domg
    flux_v     = sig_v * dv
    flux_phi   = mu_v  * dphi

    f_u     = _simp(sp.diff(u_ms, t)     - sp.diff(flux_u, s))
    f_omega = _simp(sp.diff(omega_ms, t) - sp.diff(flux_omega, s) + c_v * omega_ms - d_v * u_ms)
    f_v     = _simp(sp.diff(v_ms, t)     - sp.diff(flux_v, s)    + lam * v_ms)
    f_phi   = _simp(sp.diff(phi_ms, t)   - sp.diff(flux_phi, s)  + a_v * phi_ms - b_v * v_ms)

    return {
        'u': u_ms, 'omega': omega_ms, 'v': v_ms, 'phi': phi_ms,
        'f_u': f_u, 'f_omega': f_omega, 'f_v': f_v, 'f_phi': f_phi,
        'flux_u':     _simp(flux_u),
        'flux_omega': _simp(flux_omega),
        'flux_v':     _simp(flux_v),
        'flux_phi':   _simp(flux_phi),
    }


# ---------------------------------------------------------------------------
# TOML generation
# ---------------------------------------------------------------------------

def _ic_str(sol_expr) -> str:
    at_t0 = _simp(sol_expr.subs(t, ZERO))
    return f'"{_str(at_t0)}"'


def _bc_block(mms: Dict) -> str:
    lines = []
    for eq, flux_key in [('u', 'flux_u'), ('omega', 'flux_omega'),
                          ('v', 'flux_v'), ('phi', 'flux_phi')]:
        if _str(mms[eq]) == 'zeros':
            lines.append(f'# {eq}: exact solution zero — homogeneous Neumann by default')
            continue
        lines.append(f'B0_{eq} = {{ type = "dirichlet", data = "{_str(mms[eq])}" }}')
        lines.append(f'B1_{eq} = {{ type = "neumann",   data = "{_str(_simp(mms[flux_key]))}" }}')
    return "\n".join(lines)


def generate_toml(case: Dict[str, Any], mms: Dict, n_elements: int = 40) -> str:
    p      = case['params']
    name   = case['name']
    T_end  = case.get('T',  1.0)
    dt_val = case.get('dt', 0.025)
    chi_type = case.get('chi_type', 'constant')
    k1 = case.get('k1', 0.0);  k2 = case.get('k2', 1.0)
    m1 = case.get('m1', 0.0);  m2 = case.get('m2', 1.0)
    bc_block = _bc_block(mms)

    return textwrap.dedent(f"""\
    # MMS test: {name}
    # Auto-generated by validation_scripts/mms_ooc_upwind.py
    #
    # Manufactured solutions:
    #   u     = {mms['u']}
    #   omega = {mms['omega']}
    #   v     = {mms['v']}
    #   phi   = {mms['phi']}
    #
    # Forcing terms:
    #   f_u     = {mms['f_u']}
    #   f_omega = {mms['f_omega']}
    #   f_v     = {mms['f_v']}
    #   f_phi   = {mms['f_phi']}

    [problem]
    name = "MMS_{name}"
    neq = 4
    problem_type = "ooc"

    [time_parameters]
    T  = {T_end}
    dt = {dt_val}

    [physical_parameters.viscosity]
    nu      = {p['nu']}
    mu      = {p['mu']}
    epsilon = {p['epsilon']}
    sigma   = {p['sigma']}

    [physical_parameters.reaction]
    a = {p['a']}
    c = {p['c']}

    [physical_parameters.coupling]
    b = {p['b']}
    d = {p['d']}

    [physical_parameters.chemotaxis]
    type = "{chi_type}"
    k1   = {k1}
    k2   = {k2}

    [physical_parameters.tumor_suppression]
    m1 = {m1}
    m2 = {m2}

    [discretization]
    n_elements = {n_elements}
    tau = [0.5, 0.5, 0.5, 0.5]
    gam = 1.0   # Upwinding parameter for static condensation (0.0 = no upwind)

    [initial_conditions]
    u     = {_ic_str(mms['u'])}
    omega = {_ic_str(mms['omega'])}
    v     = {_ic_str(mms['v'])}
    phi   = {_ic_str(mms['phi'])}

    [force_functions]
    u     = "{_str(mms['f_u'])}"
    omega = "{_str(mms['f_omega'])}"
    v     = "{_str(mms['f_v'])}"
    phi   = "{_str(mms['f_phi'])}"

    [exact_solutions]
    u     = "{_str(mms['u'])}"
    omega = "{_str(mms['omega'])}"
    v     = "{_str(mms['v'])}"
    phi   = "{_str(mms['phi'])}"

    [exact_solution_derivatives]
    u     = "{_str(sp.diff(mms['u'], s))}"
    omega = "{_str(sp.diff(mms['omega'], s))}"
    v     = "{_str(sp.diff(mms['v'], s))}"
    phi   = "{_str(sp.diff(mms['phi'], s))}"

    [boundary_conditions]
    # Dirichlet at B0 (left, s=0), Neumann at B1 (right).
    # Neumann data = outward normal flux = +flux at right.
    {bc_block}
    """)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

_DECOUPLED = dict(nu=0.25, mu=0.25, epsilon=0.25, sigma=0.25, a=1.0, b=0.0, c=1.0, d=0.0)
_COUPLED   = dict(nu=0.25, mu=1.0, epsilon=1.0, sigma=1.0, a=1.0, b=1.0, c=1.0, d=1.0)

_lin  = t * s/6.28
_quad = t * (s/6.28)**2
_sin  = t * sp.sin(s) + 1.0
_psin = t * sp.sin(s + 2*t) + 1.0
_cos_t_sin_s = sp.cos(t) * sp.sin(s) + 1.0


CASES = [
    {'name': 'full_test_uncoupled', 'params': _DECOUPLED,   'u': _cos_t_sin_s, 'omega': _quad, 'v': _lin, 'phi': _psin,
     'chi_type': 'constant', 'k1': 0.0, 'k2': 2.0},
    {'name': 'full_test_weaklycoupled', 'params': _COUPLED,   'u': _cos_t_sin_s, 'omega': _quad, 'v': _lin, 'phi': _psin,
     'chi_type': 'constant', 'k1': 0.0, 'k2': 2.0},
    {'name': 'full_test_const_chi', 'params': _COUPLED,   'u': _cos_t_sin_s, 'omega': _quad, 'v': _lin, 'phi': _psin,
     'chi_type': 'constant', 'k1': 20.0, 'k2': 2.0},
    {'name': 'full_test_lin_phi', 'params': _COUPLED,   'u': _cos_t_sin_s, 'omega': _quad, 'v': _psin, 'phi': _lin,
     'chi_type': 'receptor_saturation', 'k1': 1.0, 'k2': 2.0},
    {'name': 'full_test', 'params': _COUPLED,   'u': _cos_t_sin_s, 'omega': _quad, 'v': _psin, 'phi': _sin,
     'chi_type': 'receptor_saturation', 'k1': 1.0, 'k2': 2.0},
]

for _c in CASES:
    _c.setdefault('T', 1.0);         _c.setdefault('dt', 0.025)
    _c.setdefault('chi_type', 'constant')
    _c.setdefault('k1', 0.0);        _c.setdefault('k2', 1.0)
    _c.setdefault('m1', 0.0);        _c.setdefault('m2', 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dry-run',    action='store_true', help='Print TOML to stdout, do not write files')
    parser.add_argument('--case',       metavar='NAME',      help='Generate only this case')
    parser.add_argument('--list',       action='store_true', help='List case names and exit')
    parser.add_argument('--out-dir',    default='config/mms', help='Output directory (default: config/mms)')
    parser.add_argument('--n-elements', type=int, default=40, help='Elements per domain (default: 40)')
    args = parser.parse_args()

    if args.list:
        for c in CASES:
            print(c['name'])
        return

    cases_to_run = CASES
    if args.case:
        cases_to_run = [c for c in CASES if c['name'] == args.case]
        if not cases_to_run:
            available = ', '.join(c['name'] for c in CASES)
            print(f"Error: case '{args.case}' not found.\nAvailable: {available}")
            sys.exit(1)

    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)

    for case in cases_to_run:
        name = case['name']
        print(f"  {name} ...", end=' ', flush=True)
        chi_ms = _chi_expr(
            case['chi_type'], case['k1'], case['k2'],
            case['params']['nu'], case['phi'],
        )
        mms = compute_mms(
            u_ms=case['u'], omega_ms=case['omega'],
            v_ms=case['v'], phi_ms=case['phi'],
            params=case['params'],
            chi_ms=chi_ms,
        )
        toml_str = generate_toml(case, mms, n_elements=args.n_elements)
        if args.dry_run:
            print()
            print('=' * 72)
            print(toml_str)
        else:
            path = os.path.join(args.out_dir, f'mms_{name}.toml')
            with open(path, 'w') as fh:
                fh.write(toml_str)
            print(f'-> {path}')

    if not args.dry_run:
        n = len(cases_to_run)
        print(f'\n{n} config file(s) written to {args.out_dir}/')
        print('\nRun a case with:')
        print('  python examples/upwind_example_ooc.py \\')
        print('      --arc-number 1 --arc-length 6.28 \\')
        print(f'      {args.out_dir}/mms_<name>.toml')


if __name__ == '__main__':
    main()