# BioNetFlux Code Structure Migration Guide (Version 1)

> **STATUS: COMPLETED** — This migration has been fully executed. The codebase now uses `src/bionetflux/` (not `code/ooc1d/`), all imports use `bionetflux.*`, and tests are in `tests/`. This document is retained as a historical reference of the migration process.

*Conservative migration with minimal changes - import updates only*

## Migration Strategy Overview

This guide provides a **low-risk, conservative approach** to restructure BioNetFlux with:
- ✅ **Only import statement changes**
- ✅ **Preserve all existing functionality**
- ✅ **Minimal file modifications**
- ✅ **Easy rollback capability**
- ✅ **No breaking changes to algorithms**

## Prerequisites

### Required Tools
- **Git** (version 2.25+)
- **GitHub Copilot** (optional, for automated import updates)
- **Python** (3.8+)
- **Text editor** with find-and-replace capability

### Backup Strategy
```bash
# Create complete backup
cd /Users/bertoluzza/GIT/BioNetFlux
git checkout main
git pull origin main
git tag backup-pre-migration-v1-$(date +%Y%m%d)
git push origin backup-pre-migration-v1-$(date +%Y%m%d)
```

## Phase 1: Preparation (10 minutes)

### Step 1.1: Create Migration Branch
```bash
# Create and switch to migration branch
git checkout -b structure-migration-v1
git push -u origin structure-migration-v1
```

### Step 1.2: Establish Baseline
```bash
# Ensure all current tests pass
cd code
python test_geometry.py
python test_problem.py
python test_evolution+plotting.py

# Document baseline
echo "All tests passing on $(date)" > migration_baseline.txt
git add migration_baseline.txt
git commit -m "test: Establish migration baseline"
```

### Step 1.3: Create Output Directory
```bash
# Create outputs directory (git-ignored)
mkdir -p outputs/{plots,data}
echo "*" > outputs/.gitignore
git add outputs/.gitignore
git commit -m "feat: Add organized outputs directory"
```

## Phase 2: Directory Structure Migration (15 minutes)

### Step 2.1: Create New Structure
```bash
# Create new source directory structure
mkdir -p src/bionetflux/{core,geometry,problems,solver,visualization}

# Create __init__.py files
touch src/bionetflux/__init__.py
touch src/bionetflux/{core,geometry,problems,solver,visualization}/__init__.py

# Create tests directory
mkdir -p tests
```

### Step 2.2: Move Source Files with History Preservation
```bash
# Move entire ooc1d package to bionetflux
git mv code/ooc1d src/bionetflux_temp
mv src/bionetflux_temp/* src/bionetflux/
rmdir src/bionetflux_temp

# Move setup_solver.py to src/
git mv code/setup_solver.py src/setup_solver.py

# Rename only 2 files for consistency
git mv src/bionetflux/core/static_condensation_ooc.py src/bionetflux/core/static_condensation.py

git commit -m "refactor: Move source code to new structure"
```

### Step 2.3: Move Test Files
```bash
# Move test files to dedicated directory
git mv code/test_problem.py tests/test_problem.py
git mv code/test_geometry.py tests/test_geometry.py
git mv code/test_evolution+plotting.py tests/test_evolution_plotting.py

# Remove now-empty code directory
rmdir code

git commit -m "refactor: Move tests to dedicated directory"
```

## Phase 3: Update Import Statements (20 minutes)

### Step 3.1: Create Package Initialization
```bash
# Create main package __init__.py
cat > src/bionetflux/__init__.py << 'EOF'
"""BioNetFlux: Multi-Domain Biological Network Flow Simulation Framework"""

__version__ = "1.0.0"

# Main exports for convenience
from .core.problem import Problem
from .geometry.domain_geometry import DomainGeometry, DomainInfo
from .visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter

__all__ = ["Problem", "DomainGeometry", "DomainInfo", "LeanMatplotlibPlotter"]
EOF

git add src/bionetflux/__init__.py
git commit -m "feat: Add package initialization with exports"
```

### Step 3.2: Update Imports in Source Files

#### Method A: Manual Find and Replace
```bash
# Update all Python files in src/bionetflux/
find src/bionetflux -name "*.py" -type f -exec sed -i '' 's/from ooc1d\./from bionetflux./g' {} \;
find src/bionetflux -name "*.py" -type f -exec sed -i '' 's/import ooc1d\./import bionetflux./g' {} \;

# Update specific renamed file import
find src/bionetflux -name "*.py" -type f -exec sed -i '' 's/static_condensation_ooc/static_condensation/g' {} \;
```

#### Method B: With GitHub Copilot (if available)
Open each file in VS Code and use Copilot:
```
Copilot prompt: "Update all imports in this file from 'ooc1d' to 'bionetflux'"
```

### Step 3.3: Update Imports in Test Files
```bash
# Update test files
find tests -name "*.py" -type f -exec sed -i '' 's/from ooc1d\./from bionetflux./g' {} \;
find tests -name "*.py" -type f -exec sed -i '' 's/import ooc1d\./import bionetflux./g' {} \;

# Update sys.path in test files to point to src/
find tests -name "*.py" -type f -exec sed -i '' 's|sys.path.insert(0, os.path.dirname(__file__))|sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))|g' {} \;
```

### Step 3.4: Update Examples
```bash
# Update example file
sed -i '' 's/from ooc1d\./from bionetflux./g' examples/keller_segel_example.py
sed -i '' "s|sys.path.append.*|sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))|g" examples/keller_segel_example.py
```

### Step 3.5: Update Documentation Examples
```bash
# Update import examples in documentation
find docs -name "*.md" -type f -exec sed -i '' 's/from ooc1d\./from bionetflux./g' {} \;
find docs -name "*.md" -type f -exec sed -i '' 's/import ooc1d\./import bionetflux./g' {} \;
```

```bash
git add .
git commit -m "refactor: Update all import statements to bionetflux"
```

## Phase 4: Validation (10 minutes)

### Step 4.1: Test Import Updates
```bash
# Test that imports work
cd src
python -c "from bionetflux.core.problem import Problem; print('✓ Core imports working')"
python -c "from bionetflux.geometry.domain_geometry import DomainGeometry; print('✓ Geometry imports working')"
python -c "from bionetflux.visualization.lean_matplotlib_plotter import LeanMatplotlibPlotter; print('✓ Visualization imports working')"
```

### Step 4.2: Run All Tests
```bash
# Run tests from project root
cd ..
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

python tests/test_problem.py
python tests/test_geometry.py
# python tests/test_evolution_plotting.py  # This might need setup_solver path fix
```

### Step 4.3: Test Examples
```bash
# Test example still works
python examples/keller_segel_example.py
```

### Step 4.4: Fix Any Path Issues
If tests fail due to path issues:
```bash
# Update setup_solver import in test files
sed -i '' 's/from setup_solver/from src.setup_solver/g' tests/test_evolution_plotting.py

# Or add path adjustment at top of test files
```

## Phase 5: Update .gitignore (5 minutes)

### Step 5.1: Update Ignore Patterns
```bash
# Update .gitignore for new structure
cat >> .gitignore << 'EOF'

# Output directories
outputs/

# Python cache for new structure
src/**/__pycache__/
tests/__pycache__/

# IDE files
.vscode/
.idea/
EOF

git add .gitignore
git commit -m "git: Update gitignore for new structure"
```

## Phase 6: Documentation Updates (10 minutes)

### Step 6.1: Update README
Update the README.md to reflect new structure:
```markdown
## Quick Start

```python
# Add src to Python path
import sys
sys.path.insert(0, 'src')

# Import BioNetFlux components
from bionetflux.core.problem import Problem
from bionetflux.geometry.domain_geometry import DomainGeometry
from bionetflux.problems.KS_grid_geometry import create_global_framework
```

### Step 6.2: Create Migration Summary
```bash
cat > MIGRATION_V1_SUMMARY.md << 'EOF'
# Migration V1 Summary

## Changes Made
- ✅ Renamed package: `ooc1d` → `bionetflux`
- ✅ Moved source: `code/` → `src/`
- ✅ Separated tests: `code/test_*.py` → `tests/`
- ✅ Added outputs directory: `outputs/`
- ✅ Updated all import statements
- ✅ Renamed 2 files for consistency

## Import Changes
- `from bionetflux.core.problem` → `from bionetflux.core.problem`
- `from bionetflux.geometry` → `from bionetflux.geometry`
- etc.

## No Functional Changes
- ✅ All algorithms unchanged
- ✅ All APIs preserved
- ✅ All functionality maintained

## Breaking Changes
- Import statements need updating
- File paths changed for tests/examples

## Next Steps (V2)
- Hierarchical model organization
- Advanced test categorization  
- Modern Python packaging
EOF

git add MIGRATION_V1_SUMMARY.md README.md
git commit -m "docs: Update documentation for new structure"
```

## Phase 7: Final Validation and Merge (10 minutes)

### Step 7.1: Complete Test Suite
```bash
# Final validation of all components
cd src
python -c "
import bionetflux
print(f'BioNetFlux version: {bionetflux.__version__}')
print('✓ Package loads successfully')

from bionetflux.problems.KS_grid_geometry import create_global_framework
print('✓ Problem loading works')

from bionetflux.core.problem import Problem
test_problem = Problem()
print('✓ Problem class works')
"
```

### Step 7.2: Run Integration Test
```bash
# Test the main pipeline (may need minor path adjustments)
cd ..
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
# python tests/test_evolution_plotting.py  # Fix any remaining path issues
```

### Step 7.3: Merge to Main
```bash
# If all tests pass
git checkout main
git merge structure-migration-v1
git push origin main

# Tag the migration
git tag v1.0.0-structure-migration
git push origin v1.0.0-structure-migration
```

## Troubleshooting Common Issues

### Import Errors
```bash
# If getting import errors, check:
1. PYTHONPATH includes src/: `export PYTHONPATH="${PWD}/src:${PYTHONPATH}"`
2. __init__.py files exist in all directories
3. File paths are correct in import statements
```

### Test Failures
```bash
# If tests fail:
1. Check test file paths to setup_solver
2. Verify relative imports are updated
3. Ensure all ooc1d → bionetflux replacements are complete
```

### Example Issues
```bash
# If examples don't work:
1. Check sys.path points to src/
2. Verify import statements updated
3. Test from project root directory
```

## Rollback Strategy

### Quick Rollback
```bash
# If major issues occur
git checkout main
git reset --hard backup-pre-migration-v1-$(date +%Y%m%d)
```

### Partial Rollback
```bash
# Rollback specific commits
git revert <commit-hash>
```

## Success Criteria

### ✅ Migration Complete When:
- [ ] All imports use `bionetflux` instead of `ooc1d`
- [ ] Tests run successfully from new locations
- [ ] Examples work with new structure
- [ ] Package can be imported: `import bionetflux`
- [ ] No functional changes to algorithms
- [ ] Git history preserved for all moved files
- [ ] Documentation reflects new structure

## Time Estimate: ~80 minutes total
- Phase 1 (Preparation): 10 min
- Phase 2 (Directory moves): 15 min  
- Phase 3 (Import updates): 20 min
- Phase 4 (Validation): 10 min
- Phase 5 (Gitignore): 5 min
- Phase 6 (Documentation): 10 min
- Phase 7 (Final validation): 10 min

This conservative migration approach minimizes risk while providing immediate benefits of better organization and professional naming.
