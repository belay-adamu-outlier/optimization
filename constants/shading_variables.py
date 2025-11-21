"""
Constants for shading optimization variables.

Provides a single source of truth for variable keys and bounds used by
both the optimizer setup and penalty calculations.
"""

# Ordered list of decision variable keys used in shading optimization
SHADING_VARIABLE_KEYS = [
    'south_overhang_depth',
    # 'south_overhang_width',
    'south_overhang_angle',
    'south_left_fin_depth',
    'south_right_fin_depth',
    'south_overhang_left_extension_pct',
    'south_overhang_right_extension_pct',
    'south_overhang_height_above',
    
    'west_overhang_depth',
    # 'west_overhang_width',
    'west_overhang_angle',
    'west_right_fin_depth',
    'west_left_fin_depth',
    # 'west_overhang_left_extension_pct',
    'west_overhang_right_extension_pct',
    'west_overhang_height_above',
    'east_overhang_depth',
    # 'east_overhang_width',
    'east_overhang_angle',
    'east_left_fin_depth',
    'east_right_fin_depth',
    'east_overhang_left_extension_pct',
    # 'east_overhang_right_extension_pct',
    'east_overhang_height_above',
]


# Default bounds for each decision variable
# These values are used by initialize_shading_optimization_variables and as a
# fallback when bounds are not provided at runtime.
SHADING_VARIABLE_BOUNDS = {
    'south_overhang_depth': {'min': 0.0, 'max': 2.0},
    # 'south_overhang_width': {'min': 0.0, 'max': 2.0},
    'south_overhang_angle': {'min': 0, 'max': 90.0},
    'south_left_fin_depth': {'min': 0.0, 'max': 2.0},
    'south_right_fin_depth': {'min': 0.0, 'max': 2.0},
    # 'south_overhang_left_extension_pct': {'min': 0.0, 'max': 0.5},
    # 'south_overhang_right_extension_pct': {'min': 0.0, 'max': 0.5},
    # 'south_overhang_height_above': {'min': 0.0, 'max': 1.1},
    
    'west_overhang_depth': {'min': 0.0, 'max': 2.0},
    # 'west_overhang_width': {'min': 0.0, 'max': 2.0},
    'west_overhang_angle': {'min': 0, 'max': 90.0},
    'west_right_fin_depth': {'min': 0.0, 'max': 2.0},
    'west_left_fin_depth': {'min': 0.0, 'max': 2.0},
    # 'west_overhang_left_extension_pct': {'min': 0.0, 'max': 0.5},
    # 'west_overhang_right_extension_pct': {'min': 0.0, 'max': 0.5},
    # 'west_overhang_height_above': {'min': 0.0, 'max': 1.1},
    # 'west_overhang_width': {'min': 0.0, 'max': 2.0},
    'east_overhang_depth': {'min': 0.0, 'max': 2.0},
    'east_overhang_angle': {'min': 0, 'max': 90.0},
    'east_left_fin_depth': {'min': 0.0, 'max': 2.0},
    'east_right_fin_depth': {'min': 0.0, 'max': 2.0},
    # 'east_overhang_left_extension_pct': {'min': 0.0, 'max': 0.5},
    # 'east_overhang_right_extension_pct': {'min': 0.0, 'max': 0.5},
    # 'east_overhang_height_above': {'min': 0.0, 'max': 1.1},
}


# Optional weights for penalty aggregation. Keep all as 1.0 by default.
SHADING_PENALTY_WEIGHTS = {
    'south_overhang_depth': 2.0,
    # 'south_overhang_width': 1.0,
    'south_overhang_left_extension_pct': 1.0,
    'south_overhang_right_extension_pct': 1.0,
    'south_overhang_height_above': 1.0,
    'south_overhang_angle': 3.0,
    'south_left_fin_depth': 1.0,
    'south_right_fin_depth': 1.0,
        
    'west_overhang_depth': 2.0,
    # 'west_overhang_width': 1.0,
    'west_overhang_angle': 3.0,
    'west_right_fin_depth': 1.0,
    'west_left_fin_depth': 1.0,
    'west_overhang_left_extension_pct': 1.0,
    'west_overhang_right_extension_pct': 1.0,
    'west_overhang_height_above': 1.0,
    'east_overhang_depth': 2.0,
    # 'east_overhang_width': 1.0,
    'east_overhang_angle': 3.0,
    'east_left_fin_depth': 1.0,
    'east_right_fin_depth': 1.0,
    'east_overhang_left_extension_pct': 1.0,
    'east_overhang_right_extension_pct': 1.0,
    'east_overhang_height_above': 1.0,
}


# Per-variable soft-constraint baselines (M) for penalty evaluation
#
# For each variable key, set the baseline value M used in:
#   g_j(x) = f_j(x) - M,  Q(t) = max(t, 0)^2
# If a key is not listed here, the default baseline is 0.0.
#
# Rationale:
# - Overhang angles use M=90 so angles at/near 90° are not penalized.
# - Depths, extensions, and heights keep M=0 to penalize added size.
SHADING_PENALTY_CONSTRAINTS = {
    'south_overhang_angle': 90.0,
    'west_overhang_angle': 90.0,
    'east_overhang_angle': 90.0,
}


# Extended variables for SHADING:OVERHANG and SHADING:FIN (non-breaking additions)
EXTRA_SHADING_VARIABLE_KEYS = [
    # Overhang parametrics (keep legacy width; add height_above and left/right extensions)
    'south_overhang_height_above',
    'south_overhang_left_extension_pct',
    'south_overhang_right_extension_pct',
    'west_overhang_height_above',
    'west_overhang_left_extension_pct',
    'west_overhang_right_extension_pct',
    # Fin parametrics (extensions and tilt)
    'south_left_fin_extension',
    'south_right_fin_extension',
    'south_fin_tilt',
    'west_right_fin_extension',
    'west_fin_tilt',
    'east_overhang_height_above',
    'east_overhang_left_extension_pct',
    'east_overhang_right_extension_pct',
]

# Merge keys for external use if needed
# ALL_SHADING_VARIABLE_KEYS = SHADING_VARIABLE_KEYS + EXTRA_SHADING_VARIABLE_KEYS

# Bounds for the extra variables
# EXTRA_SHADING_VARIABLE_BOUNDS = {
#     # Overhangs
#     'south_overhang_height_above': {'min': 0.0, 'max': 2.0},
#     # Overhang left/right extension as percentage of window width
#     'south_overhang_left_extension_pct': {'min': 0.0, 'max': 200.0},
#     'south_overhang_right_extension_pct': {'min': 0.0, 'max': 200.0},
#     'west_overhang_height_above': {'min': 0.0, 'max': 2.0},
#     'west_overhang_left_extension_pct': {'min': 0.0, 'max': 200.0},
#     'west_overhang_right_extension_pct': {'min': 0.0, 'max': 200.0},
#     # Fins
#     'south_left_fin_extension': {'min': 0.0, 'max': 2.0},
#     'south_right_fin_extension': {'min': 0.0, 'max': 2.0},
#     'south_fin_tilt': {'min': 0.0, 'max': 60.0},
#     'west_right_fin_extension': {'min': 0.0, 'max': 2.0},
#     'west_fin_tilt': {'min': 0.0, 'max': 60.0},
# }

# Merge bounds and expose a superset for callers that want the extended set
# SHADING_VARIABLE_BOUNDS.update(EXTRA_SHADING_VARIABLE_BOUNDS)

# Penalty weights for the extra variables (default 1.0)
# for k in EXTRA_SHADING_VARIABLE_KEYS:
#     SHADING_PENALTY_WEIGHTS.setdefault(k, 1.0)
