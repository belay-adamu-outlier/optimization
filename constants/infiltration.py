#!/usr/bin/env python3
"""
Infiltration factor constants for converting ACH at 50 Pa to design flow.

The mapping is derived from DesignBuilder exports where three factors are used:
0.072, 0.048, and 0.024. These values multiply ACH50 to obtain the effective ACH
under ambient conditions.
"""

from typing import Dict, Tuple

DEFAULT_INFILTRATION_FACTOR: float = 0.072

_ZONE_FACTOR_GROUPS: Dict[float, Tuple[str, ...]] = {
    0.072: (
        'CORRIDOR:CORRIDOR', 'CORRIDOR1:CORRIDOR', 'CORRIDOR2:CORRIDOR',
        'CORRIDOR3:ZONE1', 'E:BR', 'E:LR', 'E1:BR', 'E1:LR', 'E2:BR', 'E2:LR',
        'E3:BR', 'E3:LR', 'N:LR', 'N1:LR', 'N2:LR', 'N3:LR', 'NE:BR', 'NE:LR',
        'NE1:BR', 'NE1:LR', 'NE2:BR', 'NE2:LR', 'NE3:BR', 'NE3:LR', 'NW:BR',
        'NW:LR', 'NW1:BR', 'NW1:LR', 'NW2:BR', 'NW2:LR', 'NW3:BR', 'NW3:LR',
        'S:LR', 'S1:LR', 'S2:LR', 'S3:LR', 'SE:BR', 'SE:LR', 'SE1:BR',
        'SE1:LR', 'SE2:BR', 'SE2:LR', 'SE3:BR', 'SE3:LR', 'SW:BR', 'SW:LR',
        'SW1:BR', 'SW1:LR', 'SW2:BR', 'SW2:LR', 'SW3:BR', 'SW3:LR'
    ),
    0.048: (
        'N:BR', 'N1:BR', 'N2:BR', 'N3:BR', 'S:BR', 'S1:BR', 'S2:BR',
        'S3:BR', 'W:BR', 'W:LR', 'W1:BR', 'W1:LR', 'W2:BR', 'W2:LR',
        'W3:BR', 'W3:LR'
    ),
    0.024: ('ELEVATORANDSTAIRS2:ZONE1',),
}

INFILTRATION_ZONE_FACTORS: Dict[str, float] = {}

for factor, zones in _ZONE_FACTOR_GROUPS.items():
    for zone in zones:
        if zone in INFILTRATION_ZONE_FACTORS:
            raise ValueError(
                f'duplicate infiltration factor assignment for {zone}'
            )
        INFILTRATION_ZONE_FACTORS[zone] = factor

__all__ = [
    'DEFAULT_INFILTRATION_FACTOR',
    'INFILTRATION_ZONE_FACTORS',
]

