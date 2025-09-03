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
        'south_overhang': {'min': 0.0, 'max': 5.0},
        'south_left_fin': {'min': 0.0, 'max': 5.0},
        'south_right_fin': {'min': 0.0, 'max': 5.0},
        'west_overhang': {'min': 0.0, 'max': 5.0},
        'west_fin': {'min': 0.0, 'max': 5.0}
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


def x_to_params_envelope(x, optimization_variables):
    """Convert optimization variables x to parameter dictionary for envelope optimization.
    
    Args:
        x (array): Optimization variables array
        optimization_variables (dict): Dictionary of optimization variables and options
        
    Returns:
        dict: Parameter dictionary for envelope optimization
    """
    params = {}
    var_idx = 0
    
    for var_name, var_options in optimization_variables.items():
        if var_name.startswith('wwr_'):
            # WWR values are percentages, convert to decimal
            params[var_name] = var_options[int(x[var_idx])] / 100.0
        else:
            # Construction names
            params[var_name] = var_options[int(x[var_idx])]
        var_idx += 1
    
    return params


def x_to_params_shading(x, optimization_variables):
    """Convert optimization variables x to parameter dictionary for shading optimization.
    
    Args:
        x (array): Optimization variables array (continuous values)
        optimization_variables (dict): Dictionary of optimization variables with bounds
        
    Returns:
        dict: Parameter dictionary for shading optimization
    """
    params = {}
    var_idx = 0
    
    for var_name, bounds in optimization_variables.items():
        # x contains continuous values that are already in the correct range
        # Just ensure they're within bounds
        value = float(x[var_idx])
        value = max(bounds['min'], min(bounds['max'], value))
        params[var_name] = value
        var_idx += 1

    # Also construct nested shading dimensions for direct consumption by
    # create_shading_elements_by_orientation in modify_idf.
    dims = {}
    def ensure(side):
        if side not in dims:
            dims[side] = {}
    if 'south_overhang' in params:
        ensure('south')
        dims['south']['overhang'] = {'length': params['south_overhang']}
    if 'south_left_fin' in params:
        ensure('south')
        dims['south']['left_fin'] = {'length': params['south_left_fin']}
    if 'south_right_fin' in params:
        ensure('south')
        dims['south']['right_fin'] = {'length': params['south_right_fin']}
    if 'west_overhang' in params:
        ensure('west')
        dims['west']['overhang'] = {'length': params['west_overhang']}
    if 'west_fin' in params:
        ensure('west')
        dims['west']['right_fin'] = {'length': params['west_fin']}

    if dims:
        params['shading_dimensions'] = dims

    return params


def get_x_to_params_function(optimization_type):
    """Get the appropriate parameter conversion function based on optimization type.
    
    Args:
        optimization_type (str): Either 'envelope' or 'shading'
        
    Returns:
        function: The appropriate x_to_params function
    """
    if optimization_type.lower() == 'shading':
        return x_to_params_shading
    else:
        return x_to_params_envelope
