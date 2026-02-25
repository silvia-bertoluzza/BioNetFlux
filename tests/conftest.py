"""
Shared pytest fixtures and configuration for BioNetFlux tests.

With pip install -e . the package is on sys.path automatically.
If running without an editable install, uncomment the sys.path line below.
"""

import pytest
import numpy as np

# sys.path hack — uncomment only if NOT using pip install -e .
# import sys, os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bionetflux.utils.elementary_matrices import ElementaryMatrices


@pytest.fixture
def elementary_matrices():
    """Provide an ElementaryMatrices instance (nodal basis)."""
    return ElementaryMatrices(orthonormal_basis=False)
