"""
Problem initialization utilities for building optimization.

This module provides functions to initialize optimization problems
based on the chosen optimization type (envelope or shading).
"""




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


def initialize_shading_optimization_variables():
    """Initialize optimization variables for shading geometry optimization.
    
    Returns:
        dict: Dictionary of shading optimization variables with continuous bounds
    """
    return {
        'south_overhang_depth': {'min': 0.0, 'max': 5.0},
        'south_overhang_width': {'min': 0.0, 'max': 1.0},
        'south_overhang_angle': {'min': 0.0, 'max': 60.0},
        'south_left_fin_depth': {'min': 0.0, 'max': 5.0},
        'south_right_fin_depth': {'min': 0.0, 'max': 5.0},
        'west_overhang_depth': {'min': 0.0, 'max': 5.0},
        'west_overhang_width': {'min': 0.0, 'max': 1.0},
        'west_overhang_angle': {'min': 0.0, 'max': 60.0},
        'west_right_fin_depth': {'min': 0.0, 'max': 5.0}
    }


def get_optimization_variables(optimization_type):
    """Get optimization variables based on the chosen optimization type.
    
    Args:
        optimization_type (str): Either 'envelope' or 'shading'
        
    Returns:
        dict: Dictionary of optimization variables and their options
    """
    if optimization_type.lower() == 'shading':
        return initialize_shading_optimization_variables()
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
    
    # Additional constraints for shading optimization
    if optimization_type.lower() == 'shading':
        return base_constraints + 1  # Add total_shading_area constraint
    else:
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

    if opt == 'shading':
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
                ensure(side)
                dims[side]['overhang'] = {
                    'depth': params[f'{side}_overhang_depth'],
                    'width': params[f'{side}_overhang_width'],
                    'angle': params[f'{side}_overhang_angle']
                }
                
                if f'{side}_left_fin_depth' in params:
                    ensure(side)
                    dims[side]['left_fin'] = {'depth': params[f'{side}_left_fin_depth']}
                if f'{side}_right_fin_depth' in params:
                    ensure(side)
                    dims[side]['right_fin'] = {'depth': params[f'{side}_right_fin_depth']}

        if dims:
            params['shading_dimensions'] = dims
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
