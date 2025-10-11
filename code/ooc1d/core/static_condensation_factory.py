from typing import Type
from .static_condensation_base import StaticCondensationBase
from .static_condensation_keller_segel import KellerSegelStaticCondensation
from .static_condensation_ooc import StaticCondensationOOC
from .problem import Problem

class StaticCondensationFactory:
    """
    Factory class to create appropriate static condensation implementation
    based on problem type.
    """
    
    _implementations = {
        "keller_segel": KellerSegelStaticCondensation,
        "organ_on_chip": StaticCondensationOOC,  # Placeholder for actual OrganOnChip implementation
        # Add more implementations here as needed:
        # "reaction_diffusion": ReactionDiffusionStaticCondensation,
        # "advection_diffusion": AdvectionDiffusionStaticCondensation,
    }
    
    @classmethod
    def create(cls, problem: Problem, global_disc, elementary_matrices,
               i: int=0) -> StaticCondensationBase:
        """
        Create appropriate static condensation implementation.
        
        Args:
            problem: Problem definition with type field
            discretization: Discretization parameters
            elementary_matrices: Pre-computed elementary matrices
            
        Returns:
            Static condensation implementation for the problem type
        """
        problem_type = problem.type
        
        if problem_type not in cls._implementations:
            raise ValueError(f"Unknown problem type: {problem_type}. "
                           f"Available types: {list(cls._implementations.keys())}")
        
        implementation_class = cls._implementations[problem_type]
        return implementation_class(problem, global_disc, elementary_matrices, i)

    @classmethod
    def register_implementation(cls, problem_type: str, implementation_class: Type[StaticCondensationBase]):
        """
        Register a new static condensation implementation.
        
        Args:
            problem_type: String identifier for the problem type
            implementation_class: Class implementing StaticCondensationBase
        """
        cls._implementations[problem_type] = implementation_class
    
    @classmethod
    def get_available_types(cls):
        """Get list of available problem types."""
        return list(cls._implementations.keys())
