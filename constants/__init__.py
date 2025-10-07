#!/usr/bin/env python3
"""
Constants package for the EnergyPlus optimization project.

This package contains configuration constants organized by functionality:
- simulation_config: Simulation parameters, file paths, and shading configurations
- types: Data models and type definitions for simulations
- constraints: Optimization constraints and limits
- shading_templates: Shading element templates and configurations
- shading_variables: Optimization variables for shading parameters
"""

from .simulation_config import (
    FILE_PATHS,
    SIM_PARAMS,
    SHADING_CONFIG,
    CSV_CONFIG,
    FilePaths,
    SimulationParameters,
    ShadingConfiguration,
    CSVConfiguration
)
from .types import (
    EnvelopeCase,
    SimulationResult
)

__all__ = [
    # Simulation configuration
    'FILE_PATHS',
    'SIM_PARAMS',
    'SHADING_CONFIG',
    'CSV_CONFIG',
    'FilePaths',
    'SimulationParameters',
    'ShadingConfiguration',
    'CSVConfiguration',
    # Type definitions
    'EnvelopeCase',
    'SimulationResult'
]
