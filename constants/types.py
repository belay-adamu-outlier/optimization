#!/usr/bin/env python3
"""
Type definitions and data models for the EnergyPlus optimization project.

Contains dataclasses and type definitions used across the simulation modules.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvelopeCase:
    """Represents a single envelope case from the CSV file."""
    generation: int
    iteration: int
    external_wall: str
    roof: str
    window: str
    infiltration_ach: float
    wwr_east: float
    wwr_west: float
    wwr_north: float
    wwr_south: float
    wall_solar_absorptance: float
    roof_solar_absorptance: float
    tedi: float
    max_lr_oh: float
    max_br_oh: float


@dataclass
class SimulationResult:
    """Results from a single simulation."""
    case: EnvelopeCase
    tedi_shading: float
    max_bdrm_oh_shading: float
    max_lr_oh_shading: float
    success: bool
    error: Optional[str] = None
