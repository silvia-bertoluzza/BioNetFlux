"""
Reusable configuration management system for BioNetFlux problems.

Provides TOML-based parameter loading with function resolution and validation.
Designed to be reusable across different problem types (OoC, Keller-Segel, etc.).
"""

import numpy as np
import os
from typing import Dict, List, Optional, Callable, Any, Union
from abc import ABC, abstractmethod
import sympy as sp


def load_toml_config(config_file: str) -> Dict[str, Any]:
    """
    Load TOML configuration file with fallback for different Python versions.
    
    Args:
        config_file: Path to TOML configuration file
        
    Returns:
        Dictionary containing configuration data
        
    Raises:
        ImportError: If no TOML library available
        FileNotFoundError: If config file doesn't exist
    """
    try:
        # Python 3.11+
        import tomllib
        with open(config_file, 'rb') as f:
            return tomllib.load(f)
    except ImportError:
        try:
            # Fallback to tomli for older Python versions
            import tomli
            with open(config_file, 'rb') as f:
                return tomli.load(f)
        except ImportError:
            raise ImportError(
                "No TOML library available. Install with: pip install tomli"
            )


class FunctionResolver:
    """
    Generic function name to callable resolver with hybrid approach.
    
    Supports both named functions and string expressions using sympy.
    """
    
    def __init__(self):
        """Initialize function resolver with common function library."""
        self.function_registry = self._build_common_library()
    
    def _build_common_library(self) -> Dict[str, Callable]:
        """Build common function library available to all problems."""
        return {
            # Basic functions
            'zeros': lambda s, t=0: np.zeros_like(s),
            'ones': lambda s, t=0: np.ones_like(s),
            'constant': lambda s, t=0: np.ones_like(s),
            
            # Trigonometric functions  
            'sin_2pi': lambda s, t=0: np.sin(2 * np.pi * s),
            'cos_2pi': lambda s, t=0: np.cos(2 * np.pi * s),
            'sin_pi': lambda s, t=0: np.sin(np.pi * s),
            'cos_pi': lambda s, t=0: np.cos(np.pi * s),
            
            # Exponential and Gaussian
            'gaussian': lambda s, t=0: np.exp(-s**2),
            'exp_decay': lambda s, t=0: np.exp(-t) * np.ones_like(s),
            'exp_growth': lambda s, t=0: np.exp(t) * np.ones_like(s),
            
            # Polynomial functions
            'quadratic': lambda s, t=0: s * (1 - s),
            'cubic': lambda s, t=0: s * (1 - s) * (0.5 - s),
            
            # Step and ramp functions
            'step': lambda s, t=0: np.where(s > 0.5, 1.0, 0.0),
            'ramp': lambda s, t=0: np.maximum(0, s),
            
            # Traveling wave
            'traveling_wave_u': lambda s, t=0: ((5 * np.exp(s - t/2)) - (4 * np.exp(2*s - t)) / (np.exp(s - t/2) - 1)
            ) / (np.exp(s - t/2) - 1) - 5/8,
            'traveling_wave_phi': lambda s, t=0: 5/4 - (2 * np.exp(s - t/2)) / (np.exp(s - t/2) - 1),
            'traveling_wave_u_x': lambda s, t=0: (
                3 * np.exp(2*s - t) + 5 * np.exp(s - t/2)
            ) / (np.exp(s - t/2) - 1)**3,
            'traveling_wave_phi_x': lambda s, t=0: (
                2 * np.exp(s - t/2) * (np.exp(s - t/2) - 1) + 2 * np.exp(2*(s - t/2))
            ) / (np.exp(s - t/2) - 1)**2,   
            
        }
    
    def register_function(self, name: str, func: Callable):
        """Register a named function."""
        self.function_registry[name] = func
    
    def register_function_library(self, library: Dict[str, Callable]):
        """Register multiple functions at once."""
        self.function_registry.update(library)
    
    def parse_expression(self, expr_str: str, variables: List[str] = ['s', 't']) -> Callable:
        """
        Parse string expression into callable function using sympy.
        
        Args:
            expr_str: Mathematical expression as string
            variables: Variable names to use (default: ['s', 't'])
            
        Returns:
            Callable function
            
        Raises:
            ValueError: If expression cannot be parsed
        """
        try:
            # Create sympy symbols
            symbols = [sp.Symbol(var) for var in variables]
            
            # Parse expression
            expr = sp.sympify(expr_str)
            
            # Convert to numpy-compatible function
            func = sp.lambdify(symbols, expr, 'numpy')
            
            # Wrap to handle single argument case
            if len(variables) == 1:
                return lambda s, t=0: func(s)
            else:
                return func
                
        except Exception as e:
            raise ValueError(f"Failed to parse expression '{expr_str}': {e}")
    
    def resolve_function(self, func_spec: Union[str, Callable]) -> Callable:
        """
        Resolve function specification to callable.
        
        Args:
            func_spec: Function name (string) or direct callable
            
        Returns:
            Resolved callable function
            
        Raises:
            ValueError: If function cannot be resolved
        """
        if callable(func_spec):
            return func_spec
        
        if not isinstance(func_spec, str):
            raise ValueError(f"Function specification must be string or callable, got {type(func_spec)}")
        
        # Check if it's a registered function name
        if func_spec in self.function_registry:
            return self.function_registry[func_spec]
        
        # Try to parse as mathematical expression
        try:
            return self.parse_expression(func_spec)
        except ValueError:
            raise ValueError(f"Unknown function or invalid expression: '{func_spec}'")
    
    def resolve_function_dict(self, func_dict: Dict[str, Union[str, Callable]]) -> Dict[str, Callable]:
        """Resolve dictionary of function specifications to callables."""
        resolved = {}
        for key, func_spec in func_dict.items():
            resolved[key] = self.resolve_function(func_spec)
        return resolved


class ParameterValidator:
    """Generic parameter validation with type checking and range validation."""
    
    def __init__(self):
        """Initialize parameter validator."""
        self.validation_rules = {}
    
    def add_rule(self, param_path: str, rule: Dict[str, Any]):
        """
        Add validation rule for parameter path.
        
        Args:
            param_path: Dot-separated path to parameter (e.g., 'viscosity.nu')
            rule: Validation rule dict with keys like 'type', 'min', 'max'
        """
        self.validation_rules[param_path] = rule
    
    def validate_positive(self, value: float, param_name: str):
        """Validate parameter is positive."""
        if value <= 0:
            raise ValueError(f"Parameter '{param_name}' must be positive, got {value}")
    
    def validate_range(self, value: float, min_val: float, max_val: float, param_name: str):
        """Validate parameter is in range."""
        if not (min_val <= value <= max_val):
            raise ValueError(f"Parameter '{param_name}' must be in range [{min_val}, {max_val}], got {value}")
    
    def validate_type(self, value: Any, expected_type: type, param_name: str):
        """Validate parameter type."""
        if not isinstance(value, expected_type):
            raise ValueError(f"Parameter '{param_name}' must be of type {expected_type.__name__}, got {type(value).__name__}")
    
    def _get_nested_value(self, config: Dict[str, Any], param_path: str) -> Any:
        """Get nested parameter value from config using dot notation."""
        keys = param_path.split('.')
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate entire configuration, return list of errors.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of error messages (empty if no errors)
        """
        errors = []
        
        for param_path, rule in self.validation_rules.items():
            value = self._get_nested_value(config, param_path)
            
            if value is None:
                if rule.get('required', False):
                    errors.append(f"Required parameter '{param_path}' is missing")
                continue
            
            param_name = param_path.split('.')[-1]
            
            try:
                # Type validation
                if 'type' in rule:
                    self.validate_type(value, rule['type'], param_path)
                
                # Range validation for numeric types
                if isinstance(value, (int, float)):
                    if 'min' in rule:
                        if value < rule['min']:
                            errors.append(f"Parameter '{param_path}' must be >= {rule['min']}, got {value}")
                    if 'max' in rule:
                        if value > rule['max']:
                            errors.append(f"Parameter '{param_path}' must be <= {rule['max']}, got {value}")
                    if rule.get('positive', False):
                        self.validate_positive(value, param_path)
                
            except ValueError as e:
                errors.append(str(e))
        
        return errors


class BaseConfigManager(ABC):
    """
    Base class for problem-specific configuration management.
    
    Provides common infrastructure for TOML loading, function resolution,
    and parameter validation. Subclass this for each problem type.
    """
    
    def __init__(self, problem_type: str):
        """
        Initialize base configuration manager.
        
        Args:
            problem_type: Type identifier for the problem (e.g., 'ooc', 'keller_segel')
        """
        self.problem_type = problem_type
        self.function_resolver = FunctionResolver()
        self.validator = ParameterValidator()
        self.defaults = {}
        
        # Setup problem-specific configuration
        self._setup_defaults()
        self._setup_function_library()  
        self._setup_validation_rules()
    
    @abstractmethod
    def _setup_defaults(self):
        """Override in subclass to set problem-specific defaults."""
        pass
    
    @abstractmethod
    def _setup_function_library(self):
        """Override in subclass to set problem-specific function library."""
        pass
    
    @abstractmethod
    def _setup_validation_rules(self):
        """Override in subclass to set problem-specific validation rules."""
        pass
    
    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration with defaults."""
        def recursive_merge(default: Dict, override: Dict) -> Dict:
            result = default.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = recursive_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        return recursive_merge(self.defaults, config)
    
    def _resolve_functions(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve function specifications to callables in configuration."""
        resolved_config = config.copy()
        
        # Resolve functions in standard sections
        for section in ['initial_conditions', 'force_functions']:
            if section in resolved_config:
                resolved_config[section] = self.function_resolver.resolve_function_dict(
                    resolved_config[section]
                )
        
        return resolved_config
    
    def load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration with defaults and validation.
        
        Args:
            config_file: Path to TOML config file (optional)
            
        Returns:
            Merged and validated configuration dictionary
            
        Raises:
            ValueError: If configuration validation fails
        """
        if config_file and os.path.exists(config_file):
            print(f"Loading configuration from: {config_file}")
            config = load_toml_config(config_file)
            
            # Merge with defaults
            config = self._merge_with_defaults(config)
            
            # Validate configuration
            errors = self.validator.validate_config(config)
            if errors:
                raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
            
            print("✓ Configuration loaded and validated successfully")
        else:
            if config_file:
                print(f"Configuration file '{config_file}' not found, using defaults")
            config = self.defaults.copy()  # <-- HERE: Uses defaults when no file
    
        # Resolve functions
        config = self._resolve_functions(config)
        
        return config
    
    def get_default_config_path(self) -> str:
        """Get default configuration file path for this problem type."""
        return f"config/{self.problem_type}_parameters.toml"
    
    def _traveling_wave_u(self, s, t=0):
        """
        Keller-Segel traveling wave analytical solution for u (cell density).
        From KS_traveling_wave_double_arc.py solution_u function.
        
        Args:
            s: Spatial coordinate(s) - can be scalar or array
            t: Time coordinate - scalar
            
        Returns:
            Solution value at (s,t)
        """
        s = np.asarray(s)
        exp_st2 = np.exp(s - t/2)
        exp_2st = np.exp(2*s - t)
        
        # Avoid division by zero when exp_st2 = 1 (i.e., when s = t/2)
        denominator = exp_st2 - 1
        
        # Add small epsilon to avoid exact zero
        epsilon = 1e-15
        safe_denominator = np.where(np.abs(denominator) < epsilon, 
                                  np.sign(denominator) * epsilon, 
                                  denominator)
        
        return (5 * exp_st2) / safe_denominator - (4 * exp_2st) / (safe_denominator**2) - 5/8

    def _traveling_wave_phi(self, s, t=0):
        """
        Keller-Segel traveling wave analytical solution for phi (chemical concentration).
        From KS_traveling_wave_double_arc.py solution_phi function.
        
        Args:
            s: Spatial coordinate(s) - can be scalar or array
            t: Time coordinate - scalar
            
        Returns:
            Phi solution value at (s,t)
        """
        s = np.asarray(s)
        exp_st2 = np.exp(s - t/2)
        
        # Avoid division by zero when exp_st2 = 1 (i.e., when s = t/2)
        denominator = exp_st2 - 1
        epsilon = 1e-15
        safe_denominator = np.where(np.abs(denominator) < epsilon, 
                                  np.sign(denominator) * epsilon, 
                                  denominator)
        
        return (5*s)/4 - (5*t)/8 - 2*np.log(np.abs(safe_denominator))

    def _traveling_wave_u_x(self, s, t=0):
        """
        Spatial derivative of traveling wave u solution for chemotaxis term.
        From KS_traveling_wave_double_arc.py u_x function.
        
        Args:
            s: Spatial coordinate(s) - can be scalar or array
            t: Time coordinate - scalar
        Returns:
            Derivative value at (s,t)
        """
        s = np.asarray(s)
        exp_st2 = np.exp(s - t/2)
        exp_2st = np.exp(2*s - t)
        
        numerator = 3 * exp_2st + 5 * exp_st2
        
        # Avoid division by zero
        denominator = exp_st2 - 1
        epsilon = 1e-15
        safe_denominator = np.where(np.abs(denominator) < epsilon, 
                                  np.sign(denominator) * epsilon, 
                                  denominator)
        
        return numerator / (safe_denominator**3)

    def _traveling_wave_phi_x(self, s, t=0):
        """
        Spatial derivative of traveling wave phi solution for chemotaxis term.
        From KS_traveling_wave_double_arc.py phi_x function.
        
        Args:
            s: Spatial coordinate(s) - can be scalar or array
            t: Time coordinate - scalar
        Returns:
            Derivative value at (s,t)
        """
        s = np.asarray(s)
        exp_st2 = np.exp(s - t/2)
        
        # Avoid division by zero
        denominator = exp_st2 - 1
        epsilon = 1e-15
        safe_denominator = np.where(np.abs(denominator) < epsilon, 
                                  np.sign(denominator) * epsilon, 
                                  denominator)
        
        return 5/4 - (2 * exp_st2) / safe_denominator
