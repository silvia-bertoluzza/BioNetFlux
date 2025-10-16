"""
Geometry management module for multi-domain problems.

This module provides classes for constructing and handling complex geometries
composed of multiple domains (segments).
"""

from .domain_geometry import DomainGeometry, DomainInfo

__all__ = ['DomainGeometry', 'DomainInfo']
