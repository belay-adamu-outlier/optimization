#!/usr/bin/env python3
"""
Simulation configuration constants for envelope case simulations.

Contains file paths, thresholds, shading configurations, and other
data-related constants used in envelope case simulations.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class FilePaths:
    """File paths for simulation inputs and outputs."""
    CSV_INPUT: str = "data/AMY_envelope_optimization_final.csv"
    IDF_TEMPLATE: str = "BEMs/Building.idf"
    WEATHER_FILE: str = "BEMs/Vancouver - 2021 AMY.epw"
    IDD_PATH: str = "C:/EnergyPlusV9-4-0/Energy+.idd"


@dataclass
class SimulationParameters:
    """Simulation parameters and thresholds."""
    TEDI_THRESHOLD: float = 15.0
    DEFAULT_NUM_WORKERS: int = 4
    TEDI_CONVERSION_FACTOR: float = 2563  # kWh to TEDI conversion factor


@dataclass
class ShadingConfiguration:
    """Shading element configurations for different orientations."""

    @property
    def south_facade(self) -> Dict[str, Any]:
        """South facade: 2m overhang + 2m left fin + 2m right fin."""
        return {
            'overhang': {
                'depth': 2.0,
                'angle': 90.0
            },
            'left_fin': {'depth': 2.0},
            'right_fin': {'depth': 2.0}
        }

    @property
    def west_facade(self) -> Dict[str, Any]:
        """West facade: 2m overhang + 2m right fin only (no left fin)."""
        return {
            'overhang': {
                'depth': 2.0,
                'angle': 0.0
            },
            'left_fin': {'depth': 2.0}
        }
        
    @property
    def east_facade(self) -> Dict[str, Any]:
        """West facade: 2m overhang + 2m right fin only (no left fin)."""
        return {
            'overhang': {
                'depth': 2.0,
                'angle': 90.0
            }
        }

    @property
    def dimensions(self) -> Dict[str, Dict[str, Any]]:
        """Complete shading dimensions for all orientations."""
        return {
            'south': self.south_facade,
            'west': self.west_facade,
            'east': self.east_facade
        }


class CSVConfiguration:
    """CSV-related configuration for input and output."""

    @property
    def required_csv_columns(self) -> List[str]:
        """Required columns in the input CSV file."""
        return [
            'Thermal Energy Demand Intensity',
            'Max_Living_Room_Overheating',
            'Max_Bedroom_Overheating',
            'external_wall',
            'roof',
            'window'
        ]

    @property
    def output_fieldnames(self) -> List[str]:
        """Fieldnames for the output CSV file."""
        return [
            'Generation', 'Iteration',
            'external_wall', 'roof', 'window', 'infiltration_ach',
            'wwr_east', 'wwr_west', 'wwr_north', 'wwr_south',
            'wall_solar_absorptance', 'roof_solar_absorptance',
            'TEDI', 'Max_LR_OH', 'Max_BR_OH',
            'TEDI_Shading', 'Max_bdrm_OH_Shading', 'Max_LR_OH_Shading',
            'Success'
        ]

    @property
    def csv_metadata_keys(self) -> Dict[str, str]:
        """Metadata keys for CSV header comments."""
        return {
            'Source_File': 'Source CSV file path',
            'IDF_File': 'IDF template file path',
            'Weather_File': 'Weather file path',
            'Num_Workers': 'Number of parallel workers',
            'TEDI_Threshold': 'TEDI threshold for filtering cases',
            'Shading_Config': 'Shading configuration description',
            'Generated_At': 'Timestamp when results were generated'
        }


# Global instances
FILE_PATHS = FilePaths()
SIM_PARAMS = SimulationParameters()
SHADING_CONFIG = ShadingConfiguration()
CSV_CONFIG = CSVConfiguration()
