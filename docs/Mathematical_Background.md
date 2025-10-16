# Mathematical Background for BioNetFlux Applications

This document provides comprehensive mathematical background for the two primary application domains of BioNetFlux: Keller-Segel chemotaxis models and organ-on-chip transport systems.

## Table of Contents

1. [Keller-Segel Chemotaxis Model](#keller-segel-chemotaxis-model)
2. [Organ-on-Chip Transport Model](#organ-on-chip-transport-model)
3. [Numerical Methods](#numerical-methods)
4. [Implementation Notes](#implementation-notes)

---

## Keller-Segel Chemotaxis Model

### Overview

The Keller-Segel model describes the movement of cells in response to chemical gradients, a phenomenon known as chemotaxis. First proposed by Keller and Segel in 1970, this system has become fundamental in mathematical biology for modeling bacterial aggregation, tumor invasion, and immune cell migration.

### Governing Equations

The classical Keller-Segel system in one dimension consists of two coupled partial differential equations:

```
∂u/∂t = D_u ∂²u/∂x² - ∂/∂x[χ(u,φ) u ∂φ/∂x] + f_u(u,φ,x,t)     (1)

∂φ/∂t = D_φ ∂²φ/∂x² + g(u,φ) - δφ + f_φ(u,φ,x,t)                (2)
```

where:
- `u(x,t)` is the cell density
- `φ(x,t)` is the chemoattractant concentration
- `D_u, D_φ > 0` are diffusion coefficients
- `χ(u,φ)` is the chemotaxis sensitivity function
- `g(u,φ)` represents chemoattractant production by cells
- `δ ≥ 0` is the chemoattractant decay rate
- `f_u, f_φ` are external source terms

### Physical Interpretation

**Cell Equation (1):**
- **Diffusion term**: `D_u ∂²u/∂x²` represents random cell movement (Brownian motion)
- **Chemotactic flux**: `-∂/∂x[χ(u,φ) u ∂φ/∂x]` describes directed movement along chemical gradients
- **Source term**: `f_u(u,φ,x,t)` accounts for cell proliferation, death, or external injection

**Chemical Equation (2):**
- **Diffusion term**: `D_φ ∂²φ/∂x²` represents molecular diffusion of the chemoattractant
- **Production term**: `g(u,φ)` typically linear `g(u,φ) = αu` or saturated `g(u,φ) = αu/(1+βu)`
- **Decay term**: `-δφ` represents natural degradation
- **Source term**: `f_φ(u,φ,x,t)` accounts for external chemical sources

### Chemotaxis Sensitivity Functions

The chemotaxis sensitivity function `χ(u,φ)` is crucial for model behavior:

**1. Constant sensitivity:**
```
χ(u,φ) = χ₀ = constant
```

**2. Signal-dependent sensitivity:**
```
χ(u,φ) = χ₀/(K + φ)ⁿ
```
where `χ₀, K > 0` and `n ≥ 0` control sensitivity strength and saturation.

**3. Receptor kinetics model:**
```
χ(u,φ) = χ₀ K/(K + φ)²
```
derived from receptor binding kinetics, where `K` is the dissociation constant.

**4. Log-sensing model:**
```
χ(u,φ) = χ₀/(1 + φ)
```
captures logarithmic sensing behavior observed in bacterial chemotaxis.

### Boundary Conditions

**No-flux conditions (closed system):**
```
D_u ∂u/∂x - χ(u,φ) u ∂φ/∂x = 0    at boundaries
D_φ ∂φ/∂x = 0                      at boundaries
```

**Prescribed flux conditions:**
```
D_u ∂u/∂x - χ(u,φ) u ∂φ/∂x = j_u(t)
D_φ ∂φ/∂x = j_φ(t)
```

**Mixed conditions:**
```
u = u₀(t)        at inlet
∂u/∂x = 0        at outlet
φ = φ₀(t)        at inlet  
∂φ/∂x = 0        at outlet
```

### Analytical Solutions

For specific parameter choices, traveling wave solutions exist. Consider:
- Constant chemotaxis: `χ(u,φ) = χ₀`
- Linear chemical production: `g(u,φ) = αu`
- No decay: `δ = 0`

**Traveling wave ansatz:**
```
u(x,t) = U(ξ),    ξ = x - ct
φ(x,t) = Φ(ξ),    ξ = x - ct
```

For the parameter set `D_u = ν`, `D_φ = μ`, `χ₀ = 1/ν`, and wave speed `c = 1/2`, analytical solutions are:

```
U(ξ) = (5e^(ξ/2))/(e^(ξ/2) - 1) - (4e^ξ)/(e^(ξ/2) - 1)² - 5/8

Φ(ξ) = (5ξ)/4 - 2ln(e^(ξ/2) - 1)
```

These solutions are used in BioNetFlux for validation and testing.

### Dimensionless Analysis

**Characteristic scales:**
- Length: `L` (domain size)
- Time: `T = L²/D_u` (diffusion time)
- Cell density: `U₀` (initial density)
- Chemical concentration: `Φ₀` (reference concentration)

**Dimensionless variables:**
```
x* = x/L,  t* = t/T,  u* = u/U₀,  φ* = φ/Φ₀
```

**Dimensionless parameters:**
```
Pe_c = χ₀Φ₀L/D_u     (chemotactic Péclet number)
D = D_φ/D_u          (diffusivity ratio)  
α* = αU₀T/Φ₀         (production parameter)
δ* = δT              (decay parameter)
```

**Dimensionless equations:**
```
∂u*/∂t* = ∂²u*/∂x*² - Pe_c ∂/∂x*[χ*(u*,φ*) u* ∂φ*/∂x*] + f_u*

∂φ*/∂t* = D ∂²φ*/∂x*² + α* g*(u*,φ*) - δ* φ* + f_φ*
```

### Model Variants

**1. Volume-filling effect:**
```
∂u/∂t = ∇·[D_u(1-u/u_max)∇u - χ(φ)u(1-u/u_max)∇φ] + f_u
```

**2. Cross-diffusion:**
```
∂u/∂t = ∇·[D_u∇u + D_{uφ}u∇φ - χ(φ)u∇φ] + f_u
```

**3. Multiple species:**
```
∂u_i/∂t = D_i∇²u_i - ∇·[χ_i(φ)u_i∇φ] + R_i(u₁,...,u_n,φ)
∂φ/∂t = D_φ∇²φ + Σᵢ g_i(u_i,φ) - δφ
```

---

## Organ-on-Chip Transport Model

### Overview

Organ-on-chip (OoC) systems are microfluidic devices that simulate human organ functions. The mathematical model combines fluid flow, species transport, and cellular interactions within multi-compartment geometries.

### Governing Equations

The OoC transport model consists of coupled advection-diffusion-reaction equations:

```
∂c_i/∂t + v·∇c_i = D_i∇²c_i + R_i(c,φ) + S_i(x,t)           (3)

∂φ_j/∂t = D_φⱼ∇²φ_j + P_j(c,φ) - δ_j φ_j + Q_j(x,t)        (4)
```

where:
- `c_i(x,t)` represents concentration of species `i` (nutrients, drugs, metabolites)
- `φ_j(x,t)` represents signaling molecules or cellular markers
- `v(x)` is the fluid velocity field
- `D_i, D_φⱼ` are diffusion coefficients
- `R_i(c,φ)` represents reaction kinetics for species `i`
- `P_j(c,φ)` represents production of signaling molecules
- `δ_j` is degradation rate of signaling molecule `j`
- `S_i, Q_j` are external source/sink terms

### Multi-Compartment Structure

OoC devices typically consist of multiple interconnected compartments:

**1. Flow Channel (high flow rate, advection-dominated):**
```
∂c/∂t + v ∂c/∂x = D ∂²c/∂x² + S(x,t)

Péclet number: Pe = vL/D >> 1
```

**2. Cell Culture Chamber (low flow, reaction-dominated):**
```
∂c/∂t = D ∂²c/∂x² + R(c,φ) + S(x,t)

Damköhler number: Da = kL²/D >> 1
```

**3. Membrane Interface (selective permeability):**
```
Flux: J = P(c₁ - c₂) + σφ ∂c/∂x
```
where `P` is permeability and `σ` is reflection coefficient.

### Cellular Reaction Kinetics

**1. Michaelis-Menten kinetics:**
```
R(c) = -V_max c/(K_m + c)
```
Parameters: `V_max` (maximum rate), `K_m` (Michaelis constant)

**2. Hill kinetics (cooperative binding):**
```
R(c) = -V_max c^n/(K^n + c^n)
```
Parameters: `n` (Hill coefficient), `K` (half-saturation constant)

**3. Competitive inhibition:**
```
R(c,I) = -V_max c/[(K_m + c)(1 + I/K_i)]
```
Parameters: `I` (inhibitor concentration), `K_i` (inhibition constant)

**4. Non-competitive inhibition:**
```
R(c,I) = -V_max c/[(K_m + c)(1 + I/K_i)]
```

**5. Substrate inhibition:**
```
R(c) = -V_max c/(K_m + c + c²/K_i)
```

### Interface Conditions

At compartment interfaces, various junction conditions are implemented:

**1. Kedem-Katchalsky equations (membrane transport):**
```
J_v = L_p(ΔP - σΔπ)                    (volume flux)
J_s = ωΔπ + (1-σ)c̄J_v                  (solute flux)
```
where:
- `L_p` is hydraulic permeability
- `ω` is solute permeability  
- `σ` is reflection coefficient
- `ΔP, Δπ` are pressure and osmotic pressure differences
- `c̄` is mean concentration

**2. Simplified interface conditions:**
```
J_i = P_i(c_{i,1} - c_{i,2}) + σ_i φ ∂c_i/∂n
```

**3. Trace continuity (perfect mixing):**
```
c₁ = c₂    at interface
```

**4. Flux continuity:**
```
D₁ ∂c₁/∂n = D₂ ∂c₂/∂n    at interface
```

### Dimensionless Analysis

**Characteristic scales:**
- Length: `L` (channel length)
- Velocity: `V` (average flow velocity)
- Time: `T = L/V` (convection time)
- Concentration: `C₀` (inlet concentration)

**Dimensionless variables:**
```
x* = x/L,  t* = t/T,  c* = c/C₀,  v* = v/V
```

**Dimensionless parameters:**
```
Pe = VL/D           (Péclet number - advection vs diffusion)
Da = kL/V           (Damköhler number - reaction vs advection)  
Re = ρVL/μ          (Reynolds number - inertia vs viscosity)
Sc = μ/(ρD)         (Schmidt number - momentum vs mass diffusion)
```

**Dimensionless equation:**
```
∂c*/∂t* + Pe v*·∇*c* = ∇*²c* + Da R*(c*) + S*
```

### Typical Parameter Ranges

| Parameter | Range | Units | Application |
|-----------|-------|-------|-------------|
| Channel length | 1-10 | mm | Flow channels |
| Channel width | 10-1000 | μm | Microchannels |
| Cell chamber size | 100-5000 | μm | Culture chambers |
| Flow velocity | 0.1-10 | mm/s | Physiological flow |
| Diffusion coefficient | 10⁻¹¹-10⁻⁹ | m²/s | Small molecules |
| Permeability | 10⁻⁸-10⁻⁴ | m/s | Membrane transport |
| Reaction rate | 10⁻⁶-10⁻² | s⁻¹ | Enzymatic reactions |
| Cell density | 10⁶-10⁸ | cells/mL | Typical cultures |

### Common OoC Applications

**1. Drug screening:**
```
∂c_drug/∂t + v·∇c_drug = D∇²c_drug - k_uptake c_drug + S_inlet
∂c_metabolite/∂t = D_m∇²c_metabolite + k_metabolism c_drug - k_clearance c_metabolite
```

**2. Barrier function studies:**
```
∂c/∂t = D∇²c + R(c,TEER)
TEER dynamics: dTEER/dt = f(c_inflammatory, c_protective)
```

**3. Angiogenesis models:**
```
∂c_VEGF/∂t = D∇²c_VEGF + α_production - δc_VEGF
∂ρ_vessel/∂t = k_sprouting ∇·(ρ_vessel ∇c_VEGF) + k_growth ρ_vessel
```

---

## Numerical Methods

### Finite Element Discretization

BioNetFlux employs linear finite elements for spatial discretization. For a generic transport equation:

```
∂u/∂t = ∇·(D∇u) + v·∇u + R(u) + S
```

**Weak form:**
```
∫_Ω (∂u/∂t)w dΩ = -∫_Ω D∇u·∇w dΩ + ∫_Ω (v·∇u + R + S)w dΩ + ∫_∂Ω flux·w dΓ
```

**Matrix form:**
```
M du/dt + K u = f
```
where `M` is mass matrix, `K` is stiffness matrix, `f` is load vector.

### Time Integration

**Backward Euler (implicit):**
```
M(u^{n+1} - u^n)/Δt + K u^{n+1} = f^{n+1}
```

**Advantages:**
- Unconditionally stable
- Suitable for stiff problems
- Handles large time steps

**Crank-Nicolson (semi-implicit):**
```
M(u^{n+1} - u^n)/Δt + K(u^{n+1} + u^n)/2 = (f^{n+1} + f^n)/2
```

### Newton-Raphson Method

For nonlinear problems, Newton-Raphson iteration:

```
R(u) = M(u - u^n)/Δt + K u - f = 0
J Δu = -R(u^k)
u^{k+1} = u^k + Δu
```

where `J = ∂R/∂u` is the Jacobian matrix.

**Convergence criteria:**
```
||R(u^k)|| < tol_abs
||Δu||/||u^k|| < tol_rel
```

### Static Condensation

To reduce computational cost, interior degrees of freedom are eliminated:

```
[K_ii  K_ib] [u_i]   [f_i]
[K_bi  K_bb] [u_b] = [f_b]
```

**After condensation:**
```
K̃_bb u_b = f̃_b
K̃_bb = K_bb - K_bi K_ii^{-1} K_ib
f̃_b = f_b - K_bi K_ii^{-1} f_i
```

**Recovery:**
```
u_i = K_ii^{-1}(f_i - K_ib u_b)
```

---

## Implementation Notes

### Stability Considerations

**1. CFL condition for advection:**
```
Δt ≤ Δx/|v|
```

**2. Diffusion stability:**
```
Δt ≤ Δx²/(2D)
```

**3. Chemotaxis stability:**
```
Δt ≤ Δx²/(2χφ_max)
```

### Mesh Requirements

**1. Boundary layers:**
```
Δx_boundary ≤ √(D/v)  (for advection-diffusion)
```

**2. Reaction zones:**
```
Δx_reaction ≤ √(D/k)  (for reaction-diffusion)
```

**3. Chemotactic focusing:**
```
Δx ≤ 1/√(χ|∇φ|)
```

### Parameter Estimation

**Typical workflows:**
1. **Literature values**: Start with published parameters
2. **Dimensional analysis**: Ensure parameter scaling is correct
3. **Sensitivity analysis**: Identify critical parameters
4. **Calibration**: Fit to experimental data
5. **Validation**: Test on independent data sets

### Common Pitfalls

1. **Units consistency**: Always check dimensional analysis
2. **Mesh resolution**: Insufficient resolution can cause instabilities
3. **Time step size**: Too large steps can cause convergence issues
4. **Boundary conditions**: Incorrect BCs can dominate solution
5. **Parameter ranges**: Unphysical values can cause numerical issues

---

This mathematical background provides the theoretical foundation for understanding and implementing Keller-Segel chemotaxis and organ-on-chip transport models in BioNetFlux. For specific implementation details, refer to the main BioNetFlux documentation and example problems.
