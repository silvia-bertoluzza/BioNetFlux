# BioNetFlux Evolution Examples

This document provides quick reference commands for running the evolution examples with different problem types and configurations.

## Prerequisites

Make sure you're in the BioNetFlux root directory:
```bash
cd /Users/bertoluzza/GIT/BioNetFlux
```

## Available Evolution Examples

There are two separate evolution example scripts:
- `evolution_example_ooc.py` - For OrganOnChip problems  
- `evolution_example_ks.py` - For Keller-Segel problems

## OrganOnChip Evolution Example

### Run with default OoC configuration:
```bash
python examples/evolution_example_ooc.py config/ooc_parameters.toml
```

### Run without configuration file (uses hardcoded defaults):
```bash
python examples/evolution_example_ooc.py
```

## Keller-Segel Evolution Example

### Run with default KS configuration:
```bash
python examples/evolution_example_ks.py config/ks_parameters.toml
```

### Run with experimental KS configuration:
```bash
python examples/evolution_example_ks.py config/example_ks_experiment.toml
```

### Run without configuration file (uses hardcoded defaults):
```bash
python examples/evolution_example_ks.py
```

## Configuration Files

| Problem Type | Example Script | Compatible Config Files |
|--------------|----------------|-------------------------|
| OrganOnChip | `evolution_example_ooc.py` | `config/ooc_parameters.toml` |
| Keller-Segel | `evolution_example_ks.py` | `config/ks_parameters.toml`<br/>`config/example_ks_experiment.toml` |

## Output Files

Both examples generate:
- `geometry_with_indices.png` - Visualization of domain geometry
- `final_birdview_eq*.png` - Final solution plots for each equation
- Console output with convergence information

## Command Reference

### OrganOnChip Commands:
```bash
# Standard OoC run with 4 equations
python examples/evolution_example_ooc.py config/ooc_parameters.toml

# OoC with default settings (no config file)
python examples/evolution_example_ooc.py
```

### Keller-Segel Commands:
```bash
# Standard KS run with 2 equations
python examples/evolution_example_ks.py config/ks_parameters.toml

# KS traveling wave experiment
python examples/evolution_example_ks.py config/example_ks_experiment.toml

# KS with default settings (no config file)
python examples/evolution_example_ks.py
```

## Troubleshooting

### Script not found:
```bash
❌ Error: No such file or directory: 'examples/evolution_example.py'
```
**Solution:** Use the correct script names:
- `examples/evolution_example_ooc.py` for OrganOnChip
- `examples/evolution_example_ks.py` for Keller-Segel

### Config file not found:
```bash
❌ Error: Configuration file 'config/ks_parameters.toml' not found
```
**Solution:** Check the file path and ensure you're in the BioNetFlux root directory.

### Problem type mismatch:
```bash
❌ Configuration Error: Config file problem type mismatch
```
**Solution:** Make sure you're using the right script with the right config:
- `evolution_example_ooc.py` ↔ `ooc_parameters.toml` (problem_type = "ooc")
- `evolution_example_ks.py` ↔ `ks_parameters.toml` (problem_type = "ks")

### Function validation errors:
```bash
❌ Function validation failed: initial_conditions["u"] is not callable
```
**Solution:** Check that the configuration file uses valid function names from the function library.

## Expected Output

### Successful OrganOnChip run:
```
✓ Problem loaded: OoC_Grid_Problem
✓ Time stepper initialized  
✓ Time step successful!
🎉 Evolution example completed successfully!
```

### Successful Keller-Segel run:
```
✓ Problem loaded: KS_Double_Arc_Problem
✓ Time stepper initialized
✓ Time step successful!
🎉 Evolution example completed successfully!
```

## Problem Specifications

| Example | Equations | Default Geometry | Physics |
|---------|-----------|------------------|---------|
| OoC | 4 (u, ω, v, φ) | Grid network | Organ-on-chip flow |
| KS | 2 (u, φ) | Triple arc | Keller-Segel chemotaxis |

## Quick Test Commands

### Test both examples quickly:
```bash
# Test OoC (should show 4 equations)
python examples/evolution_example_ooc.py config/ooc_parameters.toml

# Test KS (should show 2 equations)  
python examples/evolution_example_ks.py config/ks_parameters.toml
```

### Run without config files:
```bash
# Default OoC setup
python examples/evolution_example_ooc.py

# Default KS setup
python examples/evolution_example_ks.py
```

Both should run successfully and show different equation counts and problem types in the console output.
