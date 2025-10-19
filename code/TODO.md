# BioNetFlux - TODO List

## Project Naming Decision

### 🧬 **Selected Name: BioNetFlux**
**Tagline**: "Biological Network Flux Solver"

**Why BioNetFlux is Perfect:**
- **Bio**: Clearly identifies the biological application domain
- **Net**: References network topology of 1D domains  
- **Flux**: Core mathematical concept (flux jumps, HDG methods)
- **Domain-Specific**: Directly targets biological transport phenomena
- **Professional**: Suitable for academic publications and research grants
- **Memorable**: Clear, descriptive, and brandable

### Target Applications for BioNetFlux
- **Keller-Segel Chemotaxis**: Cell migration and pattern formation
- **Vascular Networks**: Blood flow and nutrient transport  
- **Neural Networks**: Signal propagation and neurotransmitter diffusion
- **Root Systems**: Water and nutrient uptake in plants
- **Microbial Networks**: Biofilm formation and quorum sensing
- **Lymphatic Systems**: Immune cell circulation and fluid transport

### Package Structure with BioNetFlux
  - [x] Integration from trace vector (implemented)  
  - [ ] Validate against expected dual formulation behavior
- [ ] Create comprehensive tests for dual mode integration
- [ ] Document when to use dual vs primal formulation

## High Priority Issues

### 1. **Project Rebranding to BioNetFlux** (NEW)
- [ ] Update all project documentation to reflect new name
- [ ] Revise presentation materials, posters, and publications
- [ ] Communicate name change to all stakeholders

### 2. **Biological Application Development** (NEW)
- [ ] Identify and prioritize key biological applications for initial development
- [ ] Engage with biologists and domain experts for requirements gathering
- [ ] Develop proof-of-concept implementations for selected applications
- [ ] Validate model predictions with experimental or clinical data

### 3. **Advanced Numerical Methods** (NEW)
- [ ] **Add Picard iteration method for arbitrary non-linearities**
  - [ ] Design Picard iteration framework for general nonlinear problems
  - [ ] Implement fixed-point iteration schemes for nonlinear PDEs
  - [ ] Add convergence criteria and acceleration techniques
  - [ ] Support for nested nonlinearities (e.g., nonlinear diffusion + reaction)
  - [ ] Compare Picard vs Newton method performance for different problem types
  - [ ] Add hybrid Picard-Newton switching strategies
  - [ ] Test with Keller-Segel chemotaxis and other nonlinear biological models
  - [ ] Implement underrelaxation and line search for improved convergence

### 4. **Adaptive Time Stepping** (NEW)
- [ ] **Add support for adaptive time stepping**
  - [ ] Implement embedded Runge-Kutta methods for error estimation
  - [ ] Add automatic time step size control based on local truncation error
  - [ ] Support multiple time stepping strategies (constant, adaptive, event-driven)
  - [ ] Implement time step rejection and retry mechanisms
  - [ ] Add stability-based time step limiters
  - [ ] Support for maximum/minimum time step constraints
  - [ ] Create time step history tracking and analysis tools
  - [ ] Test adaptive stepping with stiff biological problems
  - [ ] Add interfaces for user-defined error tolerances and step size controls

### 5. **Non-Uniform Mesh Support** (NEW)
- [ ] **Add support for non-uniform meshes**
  - [ ] Extend Discretization class to handle variable element sizes
  - [ ] Implement adaptive mesh refinement (AMR) capabilities
  - [ ] Add mesh grading and clustering around critical regions
  - [ ] Support for user-defined mesh distribution functions
  - [ ] Update elementary matrix computations for variable element sizes
  - [ ] Modify static condensation for non-uniform element contributions
  - [ ] Add mesh quality metrics and validation
  - [ ] Implement load balancing for multi-domain non-uniform meshes
  - [ ] Create mesh generation utilities for biological applications
  - [ ] Test with problems requiring high resolution in specific regions
  - [ ] **Add find_element_containing_point to discretization module**
  - [ ] **Add summary method to discretization class**

### 6. BulkData Integration (UPDATED)
- [x] ~~Fix constructor signature mismatch in `BulkSolution.__init__()`~~ **RESOLVED: Replaced with BulkData**
- [x] ~~Implement missing methods in `BulkSolution`~~ **RESOLVED: BulkData provides comprehensive API**
- [ ] **CRITICAL**: The formula for the mass evaluation assumes the basis functions to be nodal. Extend to orthogonal basis functions
- [x] ~~Update remaining modules to use BulkData instead of BulkSolution~~ **RESOLVED: GlobalAssembler updated for lean approach**
- [x] **NEW**: Consider implementing leaner BulkDataManager that doesn't store framework objects
  - [ ] Evaluate memory usage benefits of lean approach
  - [ ] Test performance impact of passing components vs storing them
  - [ ] Document when to use lean vs full BulkDataManager
- [x] ~~**NEW - URGENT**: Fix BulkData.set_data() compatibility with bulk_by_static_condensation output~~ **RESOLVED: Direct data assignment workaround implemented**
  - [x] ~~bulk_by_static_condensation returns raw numpy arrays but BulkData.set_data() expects specific formats~~ **RESOLVED**
  - [x] ~~Need to standardize the interface between GlobalAssembler.bulk_by_static_condensation and BulkData~~ **RESOLVED**
  - [ ] Consider adding a BulkData.set_raw_data() method for direct array assignment (future enhancement)
  - [x] ~~Update time evolution loop to handle data format conversion properly~~ **RESOLVED**

### 7. **Global Assembly Enhancement** (UPDATED)
- [x] **NEW - HIGH PRIORITY**: Add treatment of non-homogeneous boundary conditions **COMPLETED**
  - [x] Implement Dirichlet boundary condition handling with prescribed values g(t) **COMPLETED**
  - [x] Add Neumann boundary condition support with prescribed fluxes q(t) **COMPLETED**
  - [x] Support time-dependent boundary data evaluation **COMPLETED**
  - [x] Integrate boundary condition treatment with constraint residual computation **COMPLETED**
  - [x] Update GlobalAssembler._add_constraint_jacobian_contributions for non-homogeneous terms **COMPLETED**
  - [x] Add validation tests for non-homogeneous boundary conditions **COMPLETED**
- [x] ~~Complete constraint Jacobian implementation~~ **IMPROVED: Better integration with BulkData**
- [ ] Update GlobalAssembler to use BulkDataManager instead of managing bulk solutions directly
- [ ] Verify global DOF mapping works with BulkData format
- [ ] Test multi-domain assembly with BulkDataManager coordination

### 8. Flux Jump Implementation (UPDATED)
- [x] ~~Fix dimension mismatch in `flux_jump.py`~~ **PARTIALLY RESOLVED: Testing infrastructure added**
- [x] ~~Test flux jump computation with lean framework approach~~ **COMPLETED: Comprehensive testing suite**
  - [x] Mock object integration validated
  - [x] Shape consistency verified across different configurations
  - [x] Mathematical properties tested (linearity, zero input handling)
  - [x] Integration with elementary matrices validated

### 9. Constraint Implementation 
  - [x] Check junction condition in T-junction double arc example - results are not convincing

## Medium Priority

### 6. Global Assembly System (UPDATED)
- [x] ~~Complete constraint Jacobian implementation~~ **IMPROVED: Better integration with BulkData**
- [ ] Update GlobalAssembler to use BulkDataManager instead of managing bulk solutions directly
- [ ] Verify global DOF mapping works with BulkData format
- [ ] Test multi-domain assembly with BulkDataManager coordination

### 7. Problem Module Consistency (UPDATED)
- [x] ~~Standardize initial condition access patterns~~ **RESOLVED: BulkDataManager handles multiple patterns**
- [x] ~~Ensure all problem modules return consistent constraint managers~~ **IMPROVED**
- [ ] Add proper forcing function evaluation testing in all problem modules
- [ ] Verify BulkDataManager extracts initial conditions and forcing functions correctly

### 8. Time Integration (UPDATED)
- [x] ~~Update time stepping implementation in `main.py` to use BulkDataManager~~ **COMPLETED: test_time_evolution.py working**
- [x] ~~Implement Newton solver using BulkDataManager forcing term computation~~ **COMPLETED: Newton solver working correctly**
- [ ] Add mass conservation tracking using BulkDataManager.compute_total_mass() during time evolution
- [x] ~~Test time stepping with single domain managed by BulkDataManager~~ **COMPLETED: Single domain working**
- [ ] **NEW**: Test time stepping with multiple domains managed by BulkDataManager
- [ ] **NEW**: Validate time evolution with boundary conditions and constraints

## Low Priority

### 9. Testing and Validation (UPDATED)
- [x] ~~Create comprehensive unit tests for each module~~ **COMPLETED: BulkData and BulkDataManager**
- [ ] Add integration tests comparing BulkData results with MATLAB reference
- [ ] Create performance benchmarks comparing BulkData vs original implementation
- [ ] Implement convergence studies using BulkDataManager

### 10. Documentation (UPDATED)
- [x] ~~Complete API documentation~~ **COMPLETED: BulkData and BulkDataManager have comprehensive docs**
- [ ] Update usage examples to use BulkData/BulkDataManager pattern
- [ ] Create migration guide from BulkSolution to BulkData
- [ ] Document dual formulation usage patterns

### 11. Performance Optimization (UPDATED)
- [ ] Profile BulkData vs BulkSolution performance
- [ ] Optimize BulkDataManager forcing term computation
- [ ] Consider caching dual BulkData objects in BulkDataManager
- [ ] Optimize element-wise mass matrix scaling in BulkData

## Code Quality Issues (UPDATED)

### 10. Code Cleanup (IMPROVED)
- [x] ~~Remove duplicate return statements~~ **RESOLVED: Fixed in BulkDataManager**
- [x] ~~Clean up unused methods and imports~~ **COMPLETED: BulkData/BulkDataManager are clean**
- [ ] **NEW**: Check for and eliminate duplications of methods from SolverSetup and lean_global_assembler
  - [ ] Identify overlapping functionality between SolverSetup and GlobalAssembler classes
  - [ ] Remove duplicate methods for initial condition creation
  - [ ] Consolidate global solution vector assembly/extraction methods  
  - [ ] Ensure single responsibility principle for each class
  - [ ] Update tests after removing duplicated methods
- [ ] **NEW**: Add better dealing with duplicate code for building jacobian and residual, and bulk solution - Code is duplicated there
  - [ ] assemble_residual_and_jacobian() and bulk_by_static_condensation() share similar validation and processing logic
  - [ ] Both methods extract trace solutions, validate forcing terms, and loop through domains
  - [ ] Consider refactoring common validation and domain iteration into helper methods
  - [ ] Could create a unified domain_processing() method that returns different outputs based on mode
- [ ] Remove or deprecate BulkSolution class once all modules migrated
- [ ] Standardize error message formats across all modules

### 13. Type Safety (IMPROVED)
- [x] ~~Add comprehensive type hints~~ **COMPLETED: BulkData and BulkDataManager fully typed**
- [ ] Add runtime type checking for BulkData.set_data() inputs
- [ ] Validate BulkDataManager input consistency at runtime

### 14. Configuration Management
- [ ] Create configuration file system leveraging BulkDataManager
- [ ] Add command-line interface using BulkDataManager for simulations
- [ ] Implement logging system with BulkDataManager integration

## Architecture Improvements (UPDATED)

### 13. Interface Standardization (IMPROVED)
- [x] ~~Define abstract base classes~~ **COMPLETED: Clean BulkData/BulkDataManager interfaces**
- [x] ~~Standardize method signatures~~ **COMPLETED: Consistent APIs**
- [ ] Update remaining modules to use new standardized interfaces
- [ ] Create adapter patterns for legacy code compatibility

### 14. Extensibility (ENHANCED)
- [ ] Design plugin system leveraging BulkDataManager domain management
- [ ] Create factory patterns using BulkDataManager.create_bulk_data()
- [ ] Add support for custom discretization schemes in BulkData
- [ ] Enable custom forcing function types in BulkDataManager

### 15. **Data Structure Abstraction** (NEW)
- [ ] **NEW**: Evaluate the possibility of defining a class TraceData to handle traces
  - [ ] Analyze current trace handling patterns across GlobalAssembler and SolverSetup
  - [ ] Design TraceData class interface for trace vector operations
  - [ ] Consider integration with domain-specific trace management
  - [ ] Evaluate benefits vs complexity of additional abstraction layer
  - [ ] Study compatibility with existing BulkData trace operations
  - [ ] Assess impact on memory efficiency and performance
  - [ ] Design factory methods for trace creation from different sources
  - [ ] Plan migration strategy from current array-based trace handling

## Recent Achievements ✅

### Lean Framework Approach Implementation
- [x] **COMPLETED**: Updated GlobalAssembler for lean BulkDataManager usage
  - [x] GlobalAssembler stores framework components directly
  - [x] BulkDataManager used only for specific operations (forcing terms, initialization)
  - [x] Clean separation of concerns between components
  - [x] Framework components accessible independently
  - [x] Newton solver integrates with direct component access
- [x] **COMPLETED**: LeanGlobalAssembler implementation
  - [x] Uses lean BulkDataManager with parameter passing approach
  - [x] Factory method for easy creation from framework objects
  - [x] Multiple initialization methods (BulkData-based and problem-based)
  - [x] Full constraint handling with external framework objects
  - [x] Comprehensive testing and validation
- [x] **COMPLETED**: Fixed lean BulkDataManager validation issue
  - [x] Added specialized single-domain validation method
  - [x] Resolved test failure in LeanGlobalAssembler
  - [x] Improved error handling for individual domain objects
  - [x] Enhanced robustness of create_bulk_data method

### Framework Integration Patterns
- [x] **COMPLETED**: Validated lean approach benefits
  - [x] Reduced coupling between BulkDataManager and other components
  - [x] More flexible framework component usage
  - [x] Easier testing and debugging of individual components
  - [x] Preparation for future memory optimization

### BulkData Class Implementation
- [x] **COMPLETED**: Flexible BulkData class with multiple initialization methods
  - [x] Direct array input (2*neq × n_elements format)
  - [x] Function-based initialization with automatic integration  
  - [x] Trace vector input with automatic reconstruction
  - [x] Dual formulation support for forcing term integration
  - [x] Comprehensive error handling and validation
  - [x] Self-testing capabilities

### BulkDataManager Implementation  
- [x] **COMPLETED**: Lean coordinator for multi-domain bulk operations
  - [x] Efficient domain data extraction and management
  - [x] Automated forcing term computation using dual BulkData integration
  - [x] Mass conservation monitoring across domains
  - [x] Comprehensive initialization and data management methods
  - [x] Extensive self-testing and validation
  - [x] Clean integration with existing HDG solver framework

### Testing and Documentation
- [x] **COMPLETED**: Comprehensive test suites for both classes
- [x] **COMPLETED**: Detailed LaTeX documentation for BulkDataManager
- [x] **COMPLETED**: Performance-oriented design with element-wise scaling

### Testing and Validation Improvements
- [x] **COMPLETED**: Added comprehensive test facility to flux_jump module
  - [x] Built-in test function with mock objects
  - [x] Multiple test scenarios (different neq and n_elements combinations)
  - [x] Edge case testing (zero inputs, boundary conditions)
  - [x] Shape validation and finite value checks
  - [x] Integration with ElementaryMatrices for realistic testing
  - [x] Consistency and linearity property validation
- [x] **COMPLETED**: Created comprehensive test script for domain_flux_jump
  - [x] Multi-suite testing approach
  - [x] Elementary matrices integration testing
  - [x] Mathematical property verification
  - [x] Detailed reporting and error analysis

### Time Evolution Implementation (NEW)
- [x] **COMPLETED**: Single domain time evolution working correctly
  - [x] Fixed bulk solution update mechanism in time loop
  - [x] Resolved BulkData compatibility issues with bulk_by_static_condensation
  - [x] Implemented proper data format conversion (extracting first 2*neq rows)
  - [x] Newton solver converging properly for linear parabolic problems
  - [x] Time-dependent source terms evaluated correctly
  - [x] Bulk solution properly updated between time steps
  - [x] Comprehensive plotting and visualization of results
  - [x] Initial vs final solution comparison working
  - [x] Analytical solution comparison (when available)

### Critical Bug Fixes (NEW)
- [x] **RESOLVED**: Stale bulk solution bug - bulk_guess now properly updated each time step
- [x] **RESOLVED**: BulkData.set_data() format incompatibility with static condensation output
- [x] **RESOLVED**: Shape mismatch in bulk data update (extracting 2*neq rows from larger arrays)
- [x] **RESOLVED**: Time inconsistency in source term evaluation and RHS assembly

## Next Steps (Immediate - UPDATED)

1. **Test multi-domain time evolution** - Extend to problems with 2+ domains and junction constraints
2. **Validate nonlinear problems** - Test with Keller-Segel chemotaxis model
3. **Performance optimization** - Profile time evolution loop for larger problems  
4. **Constraint handling** - Test time evolution with boundary conditions and junction constraints
5. **Mass conservation verification** - Add mass conservation tracking during time evolution

## Migration Strategy

### Phase 1: Lean Approach Validation (Current)
- [x] Update GlobalAssembler for lean BulkDataManager usage
- [ ] Test single and multi-domain cases with lean approach
- [ ] Validate framework component independence

### Phase 2: Performance Evaluation
- [ ] Compare lean vs full BulkDataManager approaches
- [ ] Measure memory usage differences
- [ ] Evaluate ease of use and maintainability

### Phase 3: Framework Finalization
- [ ] Choose optimal approach based on performance/usability trade-offs
- [ ] Implement final framework architecture
- [ ] Complete documentation and examples

## Architecture Notes

### Lean Approach Benefits
- **Reduced Memory Usage**: BulkDataManager doesn't store large framework objects
- **Increased Flexibility**: Framework components can be used independently
- **Better Separation**: Clear distinction between coordination and computation
- **Easier Testing**: Individual components can be tested in isolation
- **Multiple Options**: Can choose between lean and original implementations based on use case

### Implementation Comparison
- **Original GlobalAssembler**: Stores framework objects, simpler method calls
- **LeanGlobalAssembler**: External framework objects, more explicit parameter passing
- **Use Cases**: Lean for memory-constrained environments, original for simplicity

---
*Last Updated: [Current Date]*
*Status: Major architecture improvements completed - BulkData and BulkDataManager ready for production use*
3. **Medium-term**: Implement and test dual formulation
4. **Long-term**: Performance optimization and validation against MATLAB

### Notes
- BulkData provides cleaner separation of concerns than BulkSolution
- The flexible set_data interface should handle most initialization scenarios
- Consider creating factory methods for common BulkData creation patterns
