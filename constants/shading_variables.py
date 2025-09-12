"""
Constants for shading optimization variables.

Provides a single source of truth for variable keys and bounds used by
both the optimizer setup and penalty calculations.
"""

# Ordered list of decision variable keys used in shading optimization
SHADING_VARIABLE_KEYS = [
    'south_overhang_depth',
    'south_overhang_width',
    'south_overhang_angle',
    'south_left_fin_depth',
    'south_right_fin_depth',
    # 'west_overhang_depth',
    # 'west_overhang_width',
    # 'west_overhang_angle',
    # 'west_right_fin_depth',
]


# Default bounds for each decision variable
# These values are used by initialize_shading_optimization_variables and as a
# fallback when bounds are not provided at runtime.
SHADING_VARIABLE_BOUNDS = {
    'south_overhang_depth': {'min': 0.0, 'max': 2.0},
    'south_overhang_width': {'min': 0.0, 'max': 2.0},
    'south_overhang_angle': {'min': 0.0, 'max': 37.0},
    'south_left_fin_depth': {'min': 0.0, 'max': 2.0},
    'south_right_fin_depth': {'min': 0.0, 'max': 2.0},
    # 'west_overhang_depth': {'min': 0.0, 'max': 2.0},
    # 'west_overhang_width': {'min': 0.0, 'max': 2.0},
    # 'west_overhang_angle': {'min': 0.0, 'max': 37.0},
    # 'west_right_fin_depth': {'min': 0.0, 'max': 3.0},
}


# Optional weights for penalty aggregation. Keep all as 1.0 by default.
SHADING_PENALTY_WEIGHTS = {
    'south_overhang_depth': 5.0,
    'south_overhang_width': 1.0,
    'south_overhang_angle': 5.0,
    'south_left_fin_depth': 1.0,
    'south_right_fin_depth': 1.0,
    # 'west_overhang_depth': 1.0,
    # 'west_overhang_width': 1.0,
    # 'west_overhang_angle': 1.0,
    # 'west_right_fin_depth': 1.0,
}


