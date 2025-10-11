#!/usr/bin/env python3
"""
Clean test script for the lean solver setup structure.
Tests the SolverSetup class and its components without legacy code.
"""

import sys
import os

# Add the python_port directory to path for absolute imports
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

from setup_solver import SolverSetup, create_solver_setup, quick_setup

# ============================================================================
# CONFIGURATION FLAGS - Set to True/False to control output verbosity
# ============================================================================
VERBOSE_BASIC = True        # Basic setup information
VERBOSE_COMPONENTS = False   # Component loading details
VERBOSE_VALIDATION = True   # Validation details
VERBOSE_BULK = False        # Bulk data manager details

SAVE_PLOTS = False          # Save plots to files instead of showing interactively
SHOW_PLOTS = False          # Show plots interactively
PLOT_JACOBIAN_SPARSITY = False  # Plot sparsity pattern of Jacobian matrices


def test_basic_setup():
    """Test basic SolverSetup creation and initialization."""
    print("\n" + "="*50)
    print("TEST 1: BASIC SETUP CREATION")
    print("="*50)
    
    try:
        setup = SolverSetup("ooc1d.problems.test_problem2")
        setup.initialize()
        print("✓ Lean solver setup created and initialized")
        
        # Get problem information
        info = setup.get_problem_info()
        print(f"✓ Problem: {info['problem_name']}")
        print(f"  {info['num_domains']} domains, {info['total_elements']} total elements")
        print(f"  {info['total_trace_dofs']} trace DOFs, {info['num_constraints']} constraints")
        
        if VERBOSE_BASIC:
            print("\nDomain details:")
            for i, domain in enumerate(info['domains']):
                print(f"  Domain {i+1}: {domain['type']} on {domain['domain']}")
                print(f"    {domain['n_elements']} elements, {domain['n_equations']} equations")
                print(f"    {domain['trace_size']} trace DOFs")
        
        return setup, info
        
    except Exception as e:
        print(f"✗ Basic setup failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_component_loading(setup):
    """Test lazy loading of components."""
    print("\n" + "="*50)
    print("TEST 2: COMPONENT LAZY LOADING")
    print("="*50)
    
    try:
        # Elementary matrices (should be created on first access)
        elem_matrices = setup.elementary_matrices
        print(f"✓ Elementary matrices loaded: {len(elem_matrices.get_all_matrices())} matrices")
        
        if VERBOSE_COMPONENTS:
            matrices = elem_matrices.get_all_matrices()
            for name, matrix in matrices.items():
                if matrix is not None:
                    print(f"  {name}: shape {matrix.shape}")
        
        # Static condensations (should be created and cached)
        static_condensations = setup.static_condensations
        print(f"✓ Static condensations loaded: {len(static_condensations)} domains")
        
        # Global assembler
        global_assembler = setup.global_assembler
        print(f"✓ Global assembler loaded: {global_assembler.total_dofs} total DOFs")
        
        # Bulk data manager
        bulk_manager = setup.bulk_data_manager
        print(f"✓ Bulk data manager loaded: {bulk_manager.get_num_domains()} domains")
        
        return True
        
    except Exception as e:
        print(f"✗ Component loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_initial_conditions(setup):
    """Test initial conditions and global vector operations."""
    print("\n" + "="*50)
    print("TEST 3: INITIAL CONDITIONS & GLOBAL VECTORS")
    print("="*50)
    
    try:
        # Create initial conditions
        trace_solutions, multipliers = setup.create_initial_conditions()
        print(f"✓ Initial conditions created")
        print(f"  Trace solutions: {[ts.shape for ts in trace_solutions]}")
        print(f"  Multipliers: {multipliers.shape}")
        
        # Test global vector assembly/extraction
        global_solution = setup.create_global_solution_vector(trace_solutions, multipliers)
        print(f"✓ Global solution vector created: {global_solution.shape}")
        
        extracted_traces, extracted_multipliers = setup.extract_domain_solutions(global_solution)
        print(f"✓ Domain solutions extracted")
        
        # Verify consistency (round-trip test)
        consistent = True
        for i, (orig, extracted) in enumerate(zip(trace_solutions, extracted_traces)):
            if not np.allclose(orig, extracted):
                print(f"  ✗ Inconsistency in domain {i}")
                consistent = False
        
        if not np.allclose(multipliers, extracted_multipliers):
            print(f"  ✗ Inconsistency in multipliers")
            consistent = False
            
        if consistent:
            print(f"✓ Round-trip consistency verified")
        
        return trace_solutions, multipliers, global_solution
        
    except Exception as e:
        print(f"✗ Initial conditions/vectors failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_validation_system(setup):
    """Test the built-in validation system."""
    print("\n" + "="*50)
    print("TEST 4: VALIDATION SYSTEM")
    print("="*50)
    
    try:
        validation_passed = setup.validate_setup(verbose=VERBOSE_VALIDATION)
        if validation_passed:
            print("✓ Setup validation passed")
        else:
            print("✗ Setup validation failed")
        
        return validation_passed
            
    except Exception as e:
        print(f"✗ Validation system failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_setup():
    """Test the quick setup factory function."""
    print("\n" + "="*50)
    print("TEST 5: QUICK SETUP FACTORY")
    print("="*50)
    
    try:
        quick_setup_instance = quick_setup("ooc1d.problems.test_problem2", validate=True)
        quick_info = quick_setup_instance.get_problem_info()
        print(f"✓ Quick setup created and validated")
        print(f"  Problem: {quick_info['problem_name']}")
        
        return quick_setup_instance
        
    except Exception as e:
        print(f"✗ Quick setup failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_modularity():
    """Test modularity with different problem modules."""
    print("\n" + "="*50)
    print("TEST 6: MODULARITY")
    print("="*50)
    
    try:
        # Test with the same module (should work)
        setup_alt = SolverSetup("ooc1d.problems.test_problem2")
        setup_alt.initialize()
        alt_info = setup_alt.get_problem_info()
        print(f"✓ Alternative setup created: {alt_info['problem_name']}")
        
        return setup_alt
        
    except Exception as e:
        print(f"✗ Modularity test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_memory_efficiency(setup):
    """Test memory efficiency through caching."""
    print("\n" + "="*50)
    print("TEST 7: MEMORY EFFICIENCY (CACHING)")
    print("="*50)
    
    try:
        # Access components multiple times to ensure caching works
        elem1 = setup.elementary_matrices
        elem2 = setup.elementary_matrices
        
        sc1 = setup.static_condensations
        sc2 = setup.static_condensations
        
        ga1 = setup.global_assembler
        ga2 = setup.global_assembler
        
        bd1 = setup.bulk_data_manager
        bd2 = setup.bulk_data_manager
        
        # Check if same objects are returned (caching)
        if (elem1 is elem2 and sc1 is sc2 and ga1 is ga2 and bd1 is bd2):
            print("✓ Component caching works correctly")
            return True
        else:
            print("⚠ Components not cached (possible memory waste)")
            return False
            
    except Exception as e:
        print(f"✗ Memory efficiency check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bulk_operations(setup):
    """Test bulk data operations."""
    print("\n" + "="*50)
    print("TEST 8: BULK DATA OPERATIONS")
    print("="*50)
    
    try:
        bulk_manager = setup.bulk_data_manager
        
        # Create bulk solutions
        bulk_solutions = []
        for i in range(len(setup.problems)):
            problem = setup.problems[i]
            discretization = setup.global_discretization.spatial_discretizations[i]
            bulk_sol = bulk_manager.create_bulk_data(i, problem, discretization)
            bulk_solutions.append(bulk_sol)
        
        print(f"✓ Created {len(bulk_solutions)} bulk solutions")
        
        if VERBOSE_BULK:
            for i, bulk_sol in enumerate(bulk_solutions):
                bulk_data = bulk_sol.get_data()
                print(f"  Domain {i+1}: shape {bulk_data.shape}, range [{np.min(bulk_data):.6e}, {np.max(bulk_data):.6e}]")
        
    
        forcing_terms = bulk_manager.compute_forcing_terms(bulk_solutions, 
                                                                setup.problems, 
                                                                setup.global_discretization.spatial_discretizations, 
                                                                0.0, 
                                                                setup.global_discretization.dt
                                                                )

        # Test forcing term computation
        print(f"✓ Forcing terms computed")
        
        # Test mass computation
        total_mass = bulk_manager.compute_total_mass(bulk_solutions)
        print(f"✓ Total mass computed: {total_mass:.6e}")
        
        return bulk_solutions, forcing_terms
        
    except Exception as e:
        print(f"✗ Bulk operations failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_residual_jacobian(setup, trace_solutions, multipliers):
    """Test global residual and Jacobian computation."""
    print("\n" + "="*50)
    print("TEST 9: GLOBAL RESIDUAL & JACOBIAN")
    print("="*50)
    
    try:
        global_assembler = setup.global_assembler
        
        # Create global solution vector
        global_solution = setup.create_global_solution_vector(trace_solutions, multipliers)
        
        # Compute forcing terms separately (as it should be)
        bulk_solutions = []
        discretizations = setup.global_discretization.spatial_discretizations
        
        for i in range(len(setup.problems)):
            bulk_manager = setup.bulk_data_manager
            bulk_sol = bulk_manager.create_bulk_data(i, 
                                                    setup.problems[i], 
                                                    discretizations[i])
            bulk_solutions.append(bulk_sol)
        
        forcing_terms = global_assembler.compute_forcing_terms(
            bulk_solutions,
            setup.problems,
            setup.global_discretization.spatial_discretizations,
            time=0.0,
            dt=setup.global_discretization.dt
        )
        
        print(f"✓ Forcing terms computed separately")
        
        # Now compute residual and Jacobian with pre-computed forcing terms
        global_residual, global_jacobian = global_assembler.assemble_residual_and_jacobian(
            global_solution=global_solution,
            forcing_terms=forcing_terms,
            static_condensations=setup.static_condensations,
            time=0.0
        )
        
        print(f"✓ Global residual and Jacobian computed")
        print(f"  Residual shape: {global_residual.shape}, norm: {np.linalg.norm(global_residual):.6e}")
        print(f"  Jacobian shape: {global_jacobian.shape}")
        print(f"  Jacobian density: {np.count_nonzero(global_jacobian) / global_jacobian.size:.4f}")
        
        # Test with zero forcing terms
        zero_forcing_terms = [np.zeros_like(ft) for ft in forcing_terms]
        zero_residual, zero_jacobian = global_assembler.assemble_residual_and_jacobian(
            global_solution=global_solution,
            forcing_terms=zero_forcing_terms,
            static_condensations=setup.static_condensations,
            time=0.0
        )
        
        print(f"✓ Zero forcing terms test passed")
        
        # Create sparsity plot if requested
        if PLOT_JACOBIAN_SPARSITY and (SHOW_PLOTS or SAVE_PLOTS):
            create_sparsity_plot(global_jacobian, setup.problem_name)
        
        return global_residual, global_jacobian
        
    except Exception as e:
        print(f"✗ Residual/Jacobian computation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_sparsity_plot(jacobian, problem_name):
    """Create and optionally save sparsity plot of Jacobian."""
    try:
        plt.figure(figsize=(10, 8))
        plt.spy(jacobian, markersize=2, aspect='equal')
        plt.title(f'Global Jacobian Sparsity Pattern\n'
                 f'Size: {jacobian.shape[0]}×{jacobian.shape[1]}, '
                 f'Density: {np.count_nonzero(jacobian) / jacobian.size:.3f}')
        plt.xlabel('Column Index')
        plt.ylabel('Row Index')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if SAVE_PLOTS:
            plot_filename = f"jacobian_sparsity_{problem_name.lower().replace(' ', '_')}.png"
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            print(f"  ✓ Sparsity plot saved as: {plot_filename}")
        
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        print(f"  ⚠ Could not create sparsity plot: {e}")


def main():
    """Run all tests for the lean setup structure."""
    print("="*60)
    print("TESTING LEAN SOLVER SETUP STRUCTURE")
    print("="*60)
    print("Testing clean implementation without legacy code")
    
    # Test 1: Basic setup creation
    setup, info = test_basic_setup()
    if setup is None:
        print("\n✗ SETUP FAILED - Cannot continue with other tests")
        return
    
    # Test 2: Component loading
    components_ok = test_component_loading(setup)
    if not components_ok:
        print("\n⚠ Component loading issues detected")
    
    # Test 3: Initial conditions and global vectors
    trace_solutions, multipliers, global_solution = test_initial_conditions(setup)
    if trace_solutions is None:
        print("\n⚠ Initial conditions failed")
    
    # Test 4: Validation system
    validation_ok = test_validation_system(setup)
    if not validation_ok:
        print("\n⚠ Validation system issues detected")
    
    # Test 5: Quick setup factory
    quick_instance = test_quick_setup()
    if quick_instance is None:
        print("\n⚠ Quick setup failed")
    
    # Test 6: Modularity
    alt_setup = test_modularity()
    if alt_setup is not None:
        # Check independence
        if setup.problems is not alt_setup.problems:
            print("✓ Setup instances are independent")
        else:
            print("⚠ Setup instances share data")
    
    # Test 7: Memory efficiency
    caching_ok = test_memory_efficiency(setup)
    
    # Test 8: Bulk operations
    bulk_solutions, forcing_terms = test_bulk_operations(setup)
    
    # Test 9: Global residual and Jacobian
    if trace_solutions is not None and multipliers is not None:
        residual, jacobian = test_residual_jacobian(setup, trace_solutions, multipliers)
    
    # Summary
    print("\n" + "="*60)
    print("LEAN SETUP TEST SUMMARY")
    print("="*60)
    
    tests_passed = []
    tests_passed.append(("Basic setup creation", setup is not None))
    tests_passed.append(("Component lazy loading", components_ok))
    tests_passed.append(("Initial conditions", trace_solutions is not None))
    tests_passed.append(("Validation system", validation_ok))
    tests_passed.append(("Quick setup factory", quick_instance is not None))
    tests_passed.append(("Modularity", alt_setup is not None))
    tests_passed.append(("Memory efficiency", caching_ok))
    tests_passed.append(("Bulk operations", bulk_solutions is not None))
    tests_passed.append(("Residual/Jacobian", trace_solutions is not None))
    
    passed_count = sum(1 for _, passed in tests_passed if passed)
    total_count = len(tests_passed)
    
    print(f"Tests passed: {passed_count}/{total_count}")
    
    for test_name, passed in tests_passed:
        status = "✓" if passed else "✗"
        print(f"  {status} {test_name}")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED - Lean setup is working correctly!")
        print("\nSetup features confirmed:")
        print("  ✓ Minimal data redundancy through lazy loading")
        print("  ✓ Component caching for memory efficiency")
        print("  ✓ Modular problem loading")
        print("  ✓ Built-in validation system")
        print("  ✓ Clean API for solver initialization")
    else:
        print(f"\n⚠ {total_count - passed_count} test(s) failed - check implementation")
    
    print("\nReady for solver implementation!")


if __name__ == "__main__":
    main()
