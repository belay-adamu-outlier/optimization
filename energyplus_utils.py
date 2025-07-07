#!/usr/bin/env python3
"""
EnergyPlus runner with district heating and overheating analysis.
Runs an IDF file with a weather file, extracts district heating values,
calculates average overheating, and organizes output files.
"""

import os
import gc
from datetime import datetime
from eppy.modeleditor import IDF
from simulation_utils import extract_district_heating_from_htm, parse_eso_file, extract_zone_temperatures
from overheating_analysis import analyze_overheating, print_overheating_analysis_results
import numpy as np


def apply_construction_to_surfaces(idf, construction_name, surface_type):
    """
    Apply a construction to all surfaces of a specific type.
    
    Args:
        idf: IDF object
        construction_name (str): Name of the construction to apply
        surface_type (str): Type of surface ('Wall', 'Roof', 'Window')
    """
    if surface_type == 'Wall':
        surfaces = idf.idfobjects['BUILDINGSURFACE:DETAILED']
        for surface in surfaces:
            if surface.Surface_Type == 'Wall' and surface.Outside_Boundary_Condition == 'Outdoors':
                surface.Construction_Name = construction_name
    elif surface_type == 'Roof':
        surfaces = idf.idfobjects['BUILDINGSURFACE:DETAILED']
        for surface in surfaces:
            if surface.Surface_Type == 'Roof':
                surface.Construction_Name = construction_name
    elif surface_type == 'Window':
        fenestrations = idf.idfobjects['FENESTRATIONSURFACE:DETAILED']
        for fenestration in fenestrations:
            fenestration.Construction_Name = construction_name


def apply_infiltration_to_zones(idf, infiltration_ach_50pa):
    """
    Apply infiltration rate to all zones.
    
    Args:
        idf: IDF object
        infiltration_ach_50pa (float): Infiltration rate in ACH at 50 Pa
    """
    # Convert ACH at 50 Pa to ambient conditions (divide by 20 for typical conversion)
    infiltration_ach_ambient = infiltration_ach_50pa / 20.0
    
    zones = idf.idfobjects['ZONE']
    
    for zone in zones:
        zone_name = zone.Name
        
        # Check if infiltration object already exists for this zone
        existing_infiltration = None
        for infiltration in idf.idfobjects['ZONEINFILTRATION:DESIGNFLOWRATE']:
            if infiltration.Zone_or_ZoneList_Name == zone_name:
                existing_infiltration = infiltration
                break
        
        if existing_infiltration:
            # Update existing infiltration - clear Design_Flow_Rate and set Air_Changes_per_Hour
            existing_infiltration.Design_Flow_Rate = ''
            existing_infiltration.Air_Changes_per_Hour = infiltration_ach_ambient
        else:
            # Create new infiltration object
            infiltration = idf.newidfobject('ZONEINFILTRATION:DESIGNFLOWRATE')
            infiltration.Zone_or_ZoneList_Name = zone_name
            infiltration.Design_Flow_Rate = ''  # Clear this field
            infiltration.Air_Changes_per_Hour = infiltration_ach_ambient  # Set converted ACH


def calculate_window_geometry(vertices, wwr, wall_area, sill_height, wall_azimuth):
    """
    Given the original window vertices, WWR, wall area, sill height, and wall azimuth,
    return new vertices for the window that use an appropriate aspect ratio for the area,
    are horizontally centered, and maintain the sill height and wall orientation.
    The returned vertices will be in the same winding order as the input vertices.
    """
    vertices = np.array(vertices)
    wall_azimuth = wall_azimuth % 360
    # Axis-alignment check (restore original logic)
    x_vals = vertices[:, 0]
    y_vals = vertices[:, 1]
    if (not np.allclose(x_vals, x_vals[0])) and (not np.allclose(y_vals, y_vals[0])):
        raise ValueError("Window is not axis-aligned!")
    
    # Determine if wall is primarily North-South or East-West oriented
    if (45 <= wall_azimuth < 135) or (225 <= wall_azimuth < 315):
        fixed_axis = 'y'
        var_vals = vertices[:, 0]
    else:
        fixed_axis = 'x'
        var_vals = vertices[:, 1]
    
    # Calculate center of the variable coordinate (for centering the window)
    center = (np.max(var_vals) + np.min(var_vals)) / 2
    
    # Calculate new window area and dimensions
    window_area = wall_area * wwr
    if window_area < 1.5:
        aspect_ratio = 1.0
    elif window_area < 3.5:
        aspect_ratio = 1.5
    else:
        aspect_ratio = 2.0
    
    new_height = np.sqrt(window_area / aspect_ratio)
    new_width = aspect_ratio * new_height
    
    # Keep original sill height
    new_z_min = sill_height
    new_z_max = sill_height + new_height
    
    # Calculate new variable coordinate range (centered)
    new_var_min = center - new_width / 2
    new_var_max = center + new_width / 2
    
    # Build new vertices - use original fixed coordinate values directly
    if fixed_axis == 'x':
        # X is fixed, vary Y
        base_vertices = [
            (x_vals[0], new_var_min, new_z_min),
            (x_vals[0], new_var_max, new_z_min),
            (x_vals[0], new_var_max, new_z_max),
            (x_vals[0], new_var_min, new_z_max),
        ]
    else:
        # Y is fixed, vary X
        base_vertices = [
            (new_var_min, y_vals[0], new_z_min),
            (new_var_max, y_vals[0], new_z_min),
            (new_var_max, y_vals[0], new_z_max),
            (new_var_min, y_vals[0], new_z_max),
        ]
    # Map new vertices to the original winding order
    # Find the order of the original vertices (clockwise or counterclockwise)
    def polygon_area(verts):
        # Shoelace formula for area (sign gives winding)
        x = [v[0] for v in verts]
        y = [v[1] for v in verts]
        return 0.5 * sum(x[i] * y[(i+1)%4] - x[(i+1)%4] * y[i] for i in range(4))
    orig_area = polygon_area(vertices)
    base_area = polygon_area(base_vertices)
    if np.sign(orig_area) != np.sign(base_area):
        # Reverse the new vertices to match the original winding
        base_vertices = base_vertices[::-1]
    # Return new vertices in the same order as the input
    return [base_vertices[i] for i in range(4)]


def apply_wwr_to_windows(idf, wwr_east, wwr_west, wwr_north, wwr_south):
    fenestrations = idf.idfobjects['FENESTRATIONSURFACE:DETAILED']
    for fenestration in fenestrations:
        parent_wall_name = fenestration.Building_Surface_Name
        parent_wall = None
        for wall in idf.idfobjects['BUILDINGSURFACE:DETAILED']:
            if wall.Name == parent_wall_name:
                parent_wall = wall
                break
        if parent_wall is None:
            continue
        
        # Calculate azimuth from wall vertices instead of accessing non-existent field
        wall_vertices = []
        for i in range(1, 5):
            x = getattr(parent_wall, f'Vertex_{i}_Xcoordinate', None)
            y = getattr(parent_wall, f'Vertex_{i}_Ycoordinate', None)
            z = getattr(parent_wall, f'Vertex_{i}_Zcoordinate', None)
            if x is not None and y is not None and z is not None:
                wall_vertices.append((float(x), float(y), float(z)))
        
        if len(wall_vertices) < 4:
            continue
        
        # Calculate azimuth from wall orientation
        # Use the first two vertices to determine the wall direction
        v0, v1 = wall_vertices[0], wall_vertices[1]
        dx = v1[0] - v0[0]
        dy = v1[1] - v0[1]
        
        # Calculate azimuth in degrees (0 = North, 90 = East, 180 = South, 270 = West)
        # Use arctan2(dy, dx) for correct azimuth calculation
        azimuth = np.degrees(np.arctan2(dy, dx)) % 360
        
        if 45 <= azimuth < 135:
            wwr = wwr_east
        elif 135 <= azimuth < 225:
            wwr = wwr_south
        elif 225 <= azimuth < 315:
            wwr = wwr_west
        else:
            wwr = wwr_north
        
        # Calculate wall area (assume rectangle)
        v0, v1, v2, v3 = wall_vertices
        width = np.linalg.norm(np.array(v1)[:2] - np.array(v0)[:2])
        height = np.abs(v2[2] - v0[2])
        wall_area = width * height
        
        # Get window vertices
        win_vertices = []
        for i in range(1, 5):
            x = getattr(fenestration, f'Vertex_{i}_Xcoordinate', None)
            y = getattr(fenestration, f'Vertex_{i}_Ycoordinate', None)
            z = getattr(fenestration, f'Vertex_{i}_Zcoordinate', None)
            if x is not None and y is not None and z is not None:
                win_vertices.append((float(x), float(y), float(z)))
        if len(win_vertices) < 4:
            continue
        # Sill height
        sill_height = min([v[2] for v in win_vertices])
        # Calculate new vertices
        new_vertices = calculate_window_geometry(win_vertices, wwr, wall_area, sill_height, azimuth)
        # Update fenestration vertices
        for i, (x, y, z) in enumerate(new_vertices, 1):
            setattr(fenestration, f'Vertex_{i}_Xcoordinate', x)
            setattr(fenestration, f'Vertex_{i}_Ycoordinate', y)
            setattr(fenestration, f'Vertex_{i}_Zcoordinate', z)


def modify_idf(idf, envelope_parameters, save=False, save_path=None):
    """
    Apply envelope parameters to an IDF object and optionally save the modified file.
    
    Args:
        idf: IDF object to modify
        envelope_parameters (dict): Dictionary containing envelope parameters:
            - 'external_wall': External wall construction name
            - 'roof': Roof construction name
            - 'window': Window construction name
            - 'infiltration_ach': Infiltration rate in ACH at 50 Pa
            - 'wwr_east': Window-to-wall ratio for east facade
            - 'wwr_west': Window-to-wall ratio for west facade
            - 'wwr_north': Window-to-wall ratio for north facade
            - 'wwr_south': Window-to-wall ratio for south facade
        save (bool): Whether to save the modified IDF file
        save_path (str, optional): Path to save the modified IDF file. If None, uses original path
    
    Returns:
        IDF: The modified IDF object
    """
    # Apply envelope parameters
    if 'external_wall' in envelope_parameters:
        apply_construction_to_surfaces(idf, envelope_parameters['external_wall'], 'Wall')
    
    if 'roof' in envelope_parameters:
        apply_construction_to_surfaces(idf, envelope_parameters['roof'], 'Roof')
    
    if 'window' in envelope_parameters:
        apply_construction_to_surfaces(idf, envelope_parameters['window'], 'Window')
    
    # Apply infiltration
    if 'infiltration_ach' in envelope_parameters:
        apply_infiltration_to_zones(idf, envelope_parameters['infiltration_ach'])
    
    # Apply WWR
    if all(key in envelope_parameters for key in ['wwr_east', 'wwr_west', 'wwr_north', 'wwr_south']):
        apply_wwr_to_windows(idf, 
                           envelope_parameters['wwr_east'],
                           envelope_parameters['wwr_west'],
                           envelope_parameters['wwr_north'],
                           envelope_parameters['wwr_south'])
    
    # Save the modified IDF file if requested
    if save:
        if save_path is None:
            # Use the original path if no save_path provided
            save_path = idf.idfname
        idf.save(save_path)
    
    return idf


def run_simulation(idf_path, weather_path, envelope_parameters, idd_path=None, output_dir=None, verbose='v'):
    """
    Run EnergyPlus simulation with envelope parameter modifications.
    
    Args:
        idf_path (str): Path to the IDF file
        weather_path (str): Path to the weather file (.epw)
        envelope_parameters (dict): Dictionary containing envelope parameters:
            - 'external_wall': External wall construction name
            - 'roof': Roof construction name
            - 'window': Window construction name
            - 'infiltration_ach': Infiltration rate in ACH at 50 Pa
            - 'wwr_east': Window-to-wall ratio for east facade
            - 'wwr_west': Window-to-wall ratio for west facade
            - 'wwr_north': Window-to-wall ratio for north facade
            - 'wwr_south': Window-to-wall ratio for south facade
        idd_path (str, optional): Path to the IDD file
        output_dir (str, optional): Directory for output files
        verbose (str, optional): Verbosity level for EnergyPlus output:
            - 'v': Verbose (default)
            - 'q': Quiet (suppresses most logging)
            - 's': Silent (minimal output)
    
    Returns:
        str: Output directory path where simulation results are stored
    """
    idf = None
    try:
        # Set up output directory with timestamp
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"simulation_output_{timestamp}"
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set the IDD file if provided
        if idd_path:
            IDF.setiddname(idd_path)
        
        # Initialize IDF with both IDF file and weather file
        idf = IDF(idf_path, weather_path)
        
        # Apply envelope parameters using the new modify_idf function
        idf = modify_idf(idf, envelope_parameters)
        
        # Run simulation with specified output directory and verbosity
        idf.run(output_directory=output_dir, verbose=verbose)
        
        return output_dir
        
    finally:
        # Clear memory
        if idf is not None:
            del idf
        gc.collect()


def analyze_simulation_results(output_dir):
    """
    Analyze simulation results from a completed EnergyPlus simulation.
    
    Args:
        output_dir (str): Path to the simulation output directory
    
    Returns:
        dict: Dictionary containing simulation results, district heating, and overheating analysis
    """
    try:
        # Extract district heating value
        htm_file_path = os.path.join(output_dir, "eplustbl.htm")
        if os.path.exists(htm_file_path):
            district_heating = extract_district_heating_from_htm(htm_file_path)
        else:
            district_heating = None
        
        # Extract temperature data for overheating analysis
        eso_file_path = os.path.join(output_dir, "eplusout.eso")
        if os.path.exists(eso_file_path):
            temperature_data, timestamps, variable_mapping = parse_eso_file(eso_file_path)
            zone_temperatures = extract_zone_temperatures(temperature_data, timestamps)
            
            # Perform overheating analysis
            overheating_results = analyze_overheating(zone_temperatures)
            
            # Print overheating analysis results
            print_overheating_analysis_results(overheating_results)
            
            average_overheating = overheating_results['average_overheating_hours']
        else:
            average_overheating = None
            overheating_results = None
        
        results = {
            'success': True,
            'output_directory': output_dir,
            'district_heating_kwh': district_heating,
            'average_overheating_hours': average_overheating,
            'overheating_results': overheating_results
        }
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'output_directory': output_dir,
            'district_heating_kwh': None,
            'average_overheating_hours': None,
            'overheating_results': None
        }


def run_energyplus_simulation_with_analysis(idf_path, weather_path, idd_path=None, output_dir=None, verbose='v'):
    """
    Legacy function - combines run_simulation and analyze_simulation_results.
    Use the separate functions for better memory management.
    
    Args:
        idf_path (str): Path to the IDF file
        weather_path (str): Path to the weather file (.epw)
        idd_path (str, optional): Path to the IDD file. If None, eppy will try to find it automatically.
        output_dir (str, optional): Directory for output files. If None, creates a timestamped folder.
        verbose (str, optional): Verbosity level for EnergyPlus output:
            - 'v': Verbose (default)
            - 'q': Quiet (suppresses most logging)
            - 's': Silent (minimal output)
    
    Returns:
        dict: Dictionary containing simulation results, district heating, and overheating analysis
    """
    # Create empty envelope parameters for backward compatibility
    envelope_parameters = {}
    
    # Run simulation
    output_dir = run_simulation(idf_path, weather_path, envelope_parameters, idd_path, output_dir, verbose)
    
    # Analyze results
    return analyze_simulation_results(output_dir)


def print_simulation_summary(results):
    """
    Print a summary of the simulation results.
    
    Args:
        results: Results dictionary from analyze_simulation_results
    """
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    
    if results['success']:
        print(f"✓ Simulation completed successfully")
        print(f"📁 Output directory: {results['output_directory']}")
        
        if results['district_heating_kwh'] is not None:
            print(f"🔥 District Heating: {results['district_heating_kwh']:.2f} kWh")
        else:
            print("❌ District Heating: Not available")
            
        if results['average_overheating_hours'] is not None:
            print(f"🌡️  Average Overheating: {results['average_overheating_hours']:.1f} hours")
        else:
            print("❌ Average Overheating: Not available")
    else:
        print(f"❌ Simulation failed: {results.get('error', 'Unknown error')}")


# Example usage with files in current directory
if __name__ == "__main__":
    idf_file = "./Building_with_temp_output.idf"
    weather_file = "./Vancouver - 2021 AMY.epw"
    idd_file = "C:/EnergyPlusV9-4-0/Energy+.idd"
    
    # Example envelope parameters
    envelope_params = {
        'external_wall': 'R-20 Wall',
        'roof': 'R-30 Roof',
        'window': 'Double Glazed',
        'infiltration_ach': 0.5,
        'wwr_east': 0.3,
        'wwr_west': 0.3,
        'wwr_north': 0.2,
        'wwr_south': 0.4
    }
    
    # Run simulation
    output_dir = run_simulation(idf_file, weather_file, envelope_params, idd_file)
    
    # Analyze results
    results = analyze_simulation_results(output_dir)
    
    # Print summary
    print_simulation_summary(results) 