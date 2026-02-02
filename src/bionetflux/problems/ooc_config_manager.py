"""
OrganOnChip problem-specific configuration manager.
"""

import numpy as np
from typing import Dict, Any, Callable, Optional
from ..utils.config_manager import BaseConfigManager


class OoCConfigManager(BaseConfigManager):
    """OrganOnChip-specific configuration manager."""
    
    def __init__(self):
        """Initialize OoC configuration manager."""
        super().__init__("ooc")  # Config manager type for validation
    
    def _setup_defaults(self):
        """Set OoC-specific default parameters."""
        def constant_function(x):
            return np.ones_like(x)
        
        self.defaults = {
            'problem': {
                'name': 'OoC_Grid_Problem',
                'neq': 4,
                'problem_type': 'ooc'  # Config file validation type
            },
            
            'time_parameters': {
                'T': 1.0,    # Final time
                'dt': 0.1    # Time step
            },
            
            'physical_parameters': {
                'viscosity': {
                    'nu': 1.0,      # MATLAB: nu = 1.
                    'mu': 2.0,      # MATLAB: mu = 2.
                    'epsilon': 1.0, # MATLAB: epsilon = 1.
                    'sigma': 1.0    # MATLAB: sigma = 1.
                },
                'reaction': {
                    'a': 0.0,       # MATLAB: a = 0.
                    'c': 0.0        # MATLAB: c = 0.
                },
                'coupling': {
                    'b': 1.0,       # MATLAB: b = 1.
                    'd': 1.0,       # MATLAB: d = 1.
                    'chi': 1.0      # MATLAB: chi = 1.
                }
            },
            
            'discretization': {
                'n_elements': 20,
                'tau': [0.5, 0.5, 0.5, 0.5]
            },
            
            'initial_conditions': {
                'u': 'zeros',      # Default: zero
                'omega': 'zeros',  # Default: zero
                'v': 'zeros',      # Default: zero
                'phi': 'zeros',    # Default: zero
            },
            
            'force_functions': {
                'u': 'zeros',      # Default: zero force
                'omega': 'zeros',  # Default: zero force
                'v': 'zeros',      # Default: zero force
                'phi': 'zeros',    # Default: zero force
            },
            
            # New sections for domain-specific overrides (as strings, not resolved)
            'domain_initial_conditions': {},
            'domain_force_functions': {}
        }
    
    def _setup_function_library(self):
        """Set OoC-specific function library."""
        # OoC-specific functions beyond the common library
        ooc_functions = {
            # OrganOnChip specific functions
            'ooc_viscous_profile': lambda s, t=0: s * (1 - s),
            'ooc_parabolic_flow': lambda s, t=0: 4 * s * (1 - s),
            'ooc_inlet_profile': lambda s, t=0: np.where(np.abs(s) < 0.1, 1.0, 0.0),
            
            # Time-dependent functions
            'sin_t': lambda s, t=0: np.sin(t) * np.ones_like(s),
            'cos_t': lambda s, t=0: np.cos(t) * np.ones_like(s),
            'exp_t': lambda s, t=0: np.exp(-t) * np.ones_like(s),
            
            # Combined space-time functions
            'traveling_wave': lambda s, t=0: np.sin(2 * np.pi * (s - t)),
            'diffusive_wave': lambda s, t=0: np.exp(-t) * np.sin(np.pi * s),
        }
        
        self.function_resolver.register_function_library(ooc_functions)
    
    def _setup_validation_rules(self):
        """Set OoC-specific validation rules."""
        # Problem type validation
        self.validator.add_rule('problem.problem_type', 
                              {'type': str, 'required': True})
        
        # Physical parameter validation
        self.validator.add_rule('physical_parameters.viscosity.nu', 
                              {'type': float, 'min': 0.0, 'positive': True})
        self.validator.add_rule('physical_parameters.viscosity.mu', 
                              {'type': float, 'min': 0.0, 'positive': True})
        self.validator.add_rule('physical_parameters.viscosity.epsilon', 
                              {'type': float, 'min': 0.0, 'positive': True})
        self.validator.add_rule('physical_parameters.viscosity.sigma', 
                              {'type': float, 'min': 0.0, 'positive': True})
        
        # Coupling parameter validation  
        self.validator.add_rule('physical_parameters.coupling.chi', 
                              {'type': float, 'min': 0.0})
        
        # Time parameter validation
        self.validator.add_rule('time_parameters.T', 
                              {'type': float, 'min': 0.0, 'positive': True})
        self.validator.add_rule('time_parameters.dt', 
                              {'type': float, 'min': 0.0, 'positive': True})
        
        # Discretization validation
        self.validator.add_rule('discretization.n_elements', 
                              {'type': int, 'min': 1})
    
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
        
        # Validate problem type matches (for config files, we use 'ooc' as identifier)
        if config_file:  # Only validate if config file was provided
            config_problem_type = config.get('problem', {}).get('problem_type', None)
            if config_problem_type and config_problem_type != 'ooc':  # Config validation uses 'ooc'
                raise ValueError(
                    f"Configuration problem type '{config_problem_type}' does not match "
                    f"expected problem type 'ooc'. "
                    f"Please use a configuration file designed for 'ooc' problems."
                )
        
        return config
    
    def _resolve_functions(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve function specifications to callables in configuration."""
        resolved_config = config.copy()
        
        # Resolve functions in standard sections
        for section in ['initial_conditions', 'force_functions']:
            if section in resolved_config:
                section_dict = resolved_config[section].copy()
                resolved_section = {}
                
                # Resolve top-level function specifications
                for key, func_spec in section_dict.items():
                    resolved_section[key] = self.function_resolver.resolve_function(func_spec)
                
                resolved_config[section] = resolved_section
        
        # NEW: Handle domain-specific sections (keep as strings for later resolution)
        for section in ['domain_initial_conditions', 'domain_force_functions']:
            if section in resolved_config:
                # Keep domain-specific sections as strings - don't resolve them here
                # They will be resolved later in the problem module when applied
                print(f"Found {section} with {len(resolved_config[section])} entries")
                for key, value in resolved_config[section].items():
                    print(f"  {key} = {value}")
        
        return resolved_config
