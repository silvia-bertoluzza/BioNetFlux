"""
Keller-Segel problem-specific configuration manager.
"""

import numpy as np
from typing import Dict, Any, Callable, Optional
from ..utils.config_manager import BaseConfigManager


class KSConfigManager(BaseConfigManager):
    """Keller-Segel-specific configuration manager."""
    
    def __init__(self):
        """Initialize KS configuration manager."""
        super().__init__("ks")
    
    def _setup_defaults(self):
        """Set KS-specific default parameters."""
        def constant_function(x):
            return np.ones_like(x)
        
        self.defaults = {
            'problem': {
                'name': 'KS_Traveling_Wave_Problem',
                'neq': 2,
                'problem_type': 'ks'  # Config validation type
            },
            
            'time_parameters': {
                'T': 0.5,    # Final time (from KS_traveling_wave)
                'dt': 0.05   # Time step (from KS_traveling_wave)
            },
            
            'physical_parameters': {
                'diffusion': {
                    'mu': 2.0,     # Cell diffusion coefficient (from KS_traveling_wave)
                    'nu': 1.0      # Chemical diffusion coefficient (from KS_traveling_wave)
                },
                'reaction': {
                    'a': 0.0,      # Reaction parameter a (from KS_traveling_wave)
                    'b': 1.0       # Reaction parameter b (from KS_traveling_wave)
                },
                'chemotaxis': {
                    'chi': 'constant',      # Should be function name, not number
                    'dchi': 'zeros'         # Should be function name for derivative
                }  
            },
            
            'discretization': {
                'n_elements': 10,        # From KS_traveling_wave
                'tau': [1.0, 1.0]        # Stabilization parameters for 2 equations
            },
            
            'initial_conditions': {
                'u': 'zeros',      # Default: zero cell density
                'phi': 'zeros'       # Default: zero chemical concentration
            },
            
            'force_functions': {
                'u': 'zeros',      # Default: zero force
                'phi': 'zeros'       # Default: zero force
            },
            
            'exact_solutions': {
                'u': 'zeros',      # Default: zero exact solution
                'phi': 'zeros'       # Default: zero exact solution
            },
            
            'exact_solution_derivatives': {
                'u': 'zeros',      # Default: zero derivative
                'phi': 'zeros'       # Default: zero derivative
            },
            
            # Domain-specific overrides (as strings, not resolved)
            'domain_initial_conditions': {},
            'domain_force_functions': {}
        }
    
    def _setup_function_library(self):
        """Set KS-specific function library."""
        # KS-specific functions beyond the common library
        ks_functions = {
            # Keller-Segel specific functions
            'ks_gaussian_blob': lambda s, t=0: np.exp(-(s - 2.0)**2),
            'ks_step_function': lambda s, t=0: np.where(np.abs(s - 2.0) < 0.5, 1.0, 0.0),
            
            # Chemotactic patterns
            'chemotactic_gradient': lambda s, t=0: s * np.exp(-s),
            'attraction_source': lambda s, t=0: np.exp(-((s - 2.0)**2) / 0.5),
        }
        
        self.function_resolver.register_function_library(ks_functions)
    
    def _setup_validation_rules(self):
        """Set KS-specific validation rules."""
        # Problem type validation
        self.validator.add_rule('problem.problem_type', 
                              {'type': str, 'required': True})
        
        # Physical parameter validation
        self.validator.add_rule('physical_parameters.diffusion.mu', 
                              {'type': float, 'min': 0.0, 'positive': True})
        self.validator.add_rule('physical_parameters.diffusion.nu', 
                              {'type': float, 'min': 0.0, 'positive': True})
        
        # Time parameter validation
        self.validator.add_rule('time_parameters.T', 
                              {'type': float, 'min': 0.0, 'positive': True})
        self.validator.add_rule('time_parameters.dt', 
                              {'type': float, 'min': 0.0, 'positive': True})
        
        # Discretization validation
        self.validator.add_rule('discretization.n_elements', 
                              {'type': int, 'min': 1})
        
        # Physical parameter validation - chi and dchi should be function names
        self.validator.add_rule('physical_parameters.chemotaxis.chi', 
                              {'type': str, 'required': True})  # Function name
        self.validator.add_rule('physical_parameters.chemotaxis.dchi', 
                              {'type': str, 'required': True})  # Function name
    
    def load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration with problem type validation.
        
        Args:
            config_file: Path to TOML config file (optional)
            
        Returns:
            Merged and validated configuration dictionary
            
        Raises:
            ValueError: If problem type doesn't match or config validation fails
        """
        # Load using base class method
        config = super().load_config(config_file)
        
        # Validate problem type matches
        if config_file:  # Only validate if config file was provided
            config_problem_type = config.get('problem', {}).get('problem_type', None)
            if config_problem_type and config_problem_type != self.problem_type:
                raise ValueError(
                    f"Configuration problem type '{config_problem_type}' does not match "
                    f"expected problem type '{self.problem_type}'. "
                    f"Please use a configuration file designed for '{self.problem_type}' problems."
                )
        
        return config
