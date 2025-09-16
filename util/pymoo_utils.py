"""
Problem initialization utilities for building optimization.

This module provides functions to initialize optimization problems
based on the chosen optimization type (envelope or shading).
"""




from constants.shading_variables import SHADING_VARIABLE_BOUNDS
def initialize_envelope_optimization_variables():
    """Initialize optimization variables for building envelope optimization.
    
    Returns:
        dict: Dictionary of envelope optimization variables and their options
    """
    return {
        'wwr_south': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
        'wwr_west': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
        'wwr_east': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
        'wwr_north': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
        'wall_construction': ['Wall_1', 'Wall_2', 'Wall_3', 'Wall_4', 'Wall_5'],
        'roof_construction': ['Roof_1', 'Roof_2', 'Roof_3', 'Roof_4', 'Roof_5'],
        'window_construction': ['Window_1', 'Window_2', 'Window_3', 'Window_4', 'Window_5']
    }


def get_optimization_variables(optimization_type):
    """Get optimization variables based on the chosen optimization type.
    
    Args:
        optimization_type (str): Either 'envelope' or 'shading'
        
    Returns:
        dict: Dictionary of optimization variables and their options
    """
    if optimization_type.lower() == 'shading':
        return dict(SHADING_VARIABLE_BOUNDS)
    if optimization_type.lower() == 'shading_wwr':
        # Merge shading bounds with WWR floats (as decimals 0-1) for south/west
        merged = dict(SHADING_VARIABLE_BOUNDS)
        merged.update({
            'wwr_south': {'min': 0.2, 'max': 0.4},
            'wwr_west': {'min': 0.2, 'max': 0.4},
        })
        return merged
    else:
        return initialize_envelope_optimization_variables()


def get_problem_constraints(optimization_type, constrained_analysis, overheating_criterion):
    """Calculate the number of inequality constraints based on optimization type and settings.
    
    Args:
        optimization_type (str): Either 'envelope' or 'shading'
        constrained_analysis (bool): Whether constrained analysis is enabled
        overheating_criterion (str): Either 'ASHRAE' or 'CIBSE59'
        
    Returns:
        int: Number of inequality constraints
    """
    if not constrained_analysis:
        return 0
    
    # Base constraints for overheating
    # For both ASHRAE and CIBSE59, use individual room overheating values
    base_constraints = 2  # max_living_room_overheating, max_bedroom_overheating
    
    # Shading area is handled as a soft penalty on TEDI now
    return base_constraints


def get_params(optimization_type, x, optimization_variables):
    """Convert optimization vector ``x`` into a parameters dict.

    - For 'shading', returns flat shading keys and nested 'shading_dimensions'.
    - For 'envelope', returns WWRs (as decimals) and construction names.

    Args:
        optimization_type (str): 'envelope' or 'shading'
        x (array-like): Optimization decision vector
        optimization_variables (dict): Variable definitions/options/bounds

    Returns:
        dict: Parameters suitable for modify_idf
    """
    opt = optimization_type.lower()
    params: dict = {}

    if opt in ('shading', 'shading_wwr'):
        # Continuous variables within bounds
        var_idx = 0
        for var_name, bounds in optimization_variables.items():
            value = float(x[var_idx])
            value = max(bounds['min'], min(bounds['max'], value))
            params[var_name] = value
            var_idx += 1

        # Build nested shading dimensions for geometry creation
        dims: dict = {}
        
        def ensure(side: str):
            if side not in dims:
                dims[side] = {}
        
        for side in ['south', 'west']:
            # Only create blocks for sides where variables were provided
            has_overhang = (f'{side}_overhang_depth' in params)
            if has_overhang:
                ensure(side)
                dims[side]['overhang'] = {
                    'depth': params.get(f'{side}_overhang_depth'),
                    'width': params.get(f'{side}_overhang_width'),
                    'angle': params.get(f'{side}_overhang_angle'),
                    'height_above': params.get(f'{side}_overhang_height_above'),
                    # New: percentage of window width for left/right extensions
                    'left_extension_pct': params.get(f'{side}_overhang_left_extension_pct'),
                    'right_extension_pct': params.get(f'{side}_overhang_right_extension_pct'),
                }

            if f'{side}_left_fin_depth' in params:
                ensure(side)
                dims[side]['left_fin'] = {'depth': params[f'{side}_left_fin_depth']}
                if f'{side}_left_fin_extension' in params:
                    dims[side]['left_fin']['extension'] = params[f'{side}_left_fin_extension']
                if f'{side}_fin_tilt' in params:
                    dims[side]['left_fin']['tilt'] = params[f'{side}_fin_tilt']
            if f'{side}_right_fin_depth' in params:
                ensure(side)
                dims[side]['right_fin'] = {'depth': params[f'{side}_right_fin_depth']}
                if f'{side}_right_fin_extension' in params:
                    dims[side]['right_fin']['extension'] = params[f'{side}_right_fin_extension']
                if f'{side}_fin_tilt' in params:
                    dims[side]['right_fin']['tilt'] = params[f'{side}_fin_tilt']

        if dims:
            params['shading_dimensions'] = dims

        # If this is the hybrid shading_wwr, ensure WWRs are decimal floats
        if opt == 'shading_wwr':
            if 'wwr_south' in params:
                params['wwr_south'] = float(params['wwr_south'])
            if 'wwr_west' in params:
                params['wwr_west'] = float(params['wwr_west'])
        return params

    # Envelope: categorical indexes and WWRs
    var_idx = 0
    for var_name, var_options in optimization_variables.items():
        if var_name.startswith('wwr_'):
            params[var_name] = var_options[int(x[var_idx])] / 100.0
        else:
            params[var_name] = var_options[int(x[var_idx])]
        var_idx += 1

    # Duplicate construction keys to names used by modify_idf
    if 'wall_construction' in params:
        params['external_wall'] = params['wall_construction']
    if 'roof_construction' in params:
        params['roof'] = params['roof_construction']
    if 'window_construction' in params:
        params['window'] = params['window_construction']

    return params





