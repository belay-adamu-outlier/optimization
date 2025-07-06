import os
from typing import Dict, List, Tuple, Any, Generator, Optional
from eppy.modeleditor import IDF
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import re

def initialize_idf(path_to_idf: str, path_to_idd: str, weather_file: str):
    """
    Initialize an IDF object with the specified IDF file, IDD file, and weather file.
    
    Args:
        path_to_idf: Path to the IDF file
        path_to_idd: Path to the IDD file
        weather_file: Path to the weather file
    
    Returns:
        IDF object initialized with the specified files
    """
    # Set the IDD file
    IDF.setiddname(path_to_idd)
    
    # Create IDF object with IDF file and weather file
    idf = IDF(path_to_idf, weather_file)
    
    return idf


def extract_building_parameters(parameters_to_be_extracted: List[str], idf) -> Dict[str, Any]:
    """
    Extract all objects from the parameters_to_be_extracted list from the IDF.
    
    Args:
        parameters_to_be_extracted: List of parameter names to extract
        idf: IDF object to extract parameters from
    
    Returns:
        Dictionary containing the extracted parameters and their values
    """
    extracted_parameters = {}
    
    for param_name in parameters_to_be_extracted:
        try:
            # Get all objects of this type from the IDF
            objects = idf.idfobjects[param_name]
            
            if objects:
                # Store all objects of this type
                extracted_parameters[param_name] = objects
            else:
                # Store empty list if no objects found
                extracted_parameters[param_name] = []
                
        except KeyError:
            # Parameter type not found in IDF
            extracted_parameters[param_name] = []
    
    return extracted_parameters


def modify_model(idf, modifications: Dict[str, Any]):
    """
    Iterate through the modifications and modify the IDF model.
    
    Args:
        idf: IDF object to modify
        modifications: Dictionary containing modifications to apply
    
    Returns:
        Modified IDF object
    """
    # Modify the original IDF directly
    modified_idf = idf
    
    for modification_type, modification_data in modifications.items():
        if modification_type == "json_modifications":
            # Handle JSON-style modifications
            from eppy import json_functions
            json_functions.updateidf(modified_idf, modification_data)
            
        elif modification_type == "object_modifications":
            # Handle direct object modifications
            for object_type, object_modifications in modification_data.items():
                if object_type in modified_idf.idfobjects:
                    for obj_name, field_modifications in object_modifications.items():
                        # Find the object by name
                        for obj in modified_idf.idfobjects[object_type]:
                            if hasattr(obj, 'Name') and obj.Name == obj_name:
                                # Apply field modifications
                                for field_name, new_value in field_modifications.items():
                                    if hasattr(obj, field_name):
                                        setattr(obj, field_name, new_value)
                                break
                                
        elif modification_type == "add_objects":
            # Handle adding new objects
            for object_type, new_objects in modification_data.items():
                for new_object_data in new_objects:
                    # Create new object and add to IDF
                    new_obj = modified_idf.newidfobject(object_type)
                    for field_name, value in new_object_data.items():
                        if hasattr(new_obj, field_name):
                            setattr(new_obj, field_name, value)
                            
        elif modification_type == "remove_objects":
            # Handle removing objects
            for object_type, object_names in modification_data.items():
                if object_type in modified_idf.idfobjects:
                    objects_to_remove = []
                    for obj in modified_idf.idfobjects[object_type]:
                        if hasattr(obj, 'Name') and obj.Name in object_names:
                            objects_to_remove.append(obj)
                    
                    # Remove the objects
                    for obj in objects_to_remove:
                        modified_idf.removeidfobject(obj)
    
    return modified_idf


def create_simulation_scenarios(path_to_idf: str, path_to_idd: str, weather_file: str, 
                               modification_list: List[Dict[str, Any]]) -> Generator[Tuple[Any, Dict[str, Any]], None, None]:
    """
    Iterate through modification_list and using generators returns a tuple of modify_model functions
    that will be called when scenario is needed.
    
    Args:
        path_to_idf: Path to the IDF file
        path_to_idd: Path to the IDD file
        weather_file: Path to the weather file
        modification_list: List of modification dictionaries
    
    Yields:
        Tuple of (modify_model_function, scenario_data) where modify_model_function
        is a callable that will modify the IDF when called
    """
    for i, modifications in enumerate(modification_list):
        def create_modify_function(mods, scenario_id=i):
            def modify_function():
                # Initialize IDF
                idf = initialize_idf(path_to_idf, path_to_idd, weather_file)
                # Apply modifications
                modified_idf = modify_model(idf, mods)
                return modified_idf, scenario_id
            return modify_function
        
        scenario_data = {
            'scenario_id': i,
            'modifications': modifications,
            'description': modifications.get('description', f'Scenario {i}')
        }
        
        yield create_modify_function(modifications), scenario_data


def get_simulation_result(idf, results: List[str]) -> Dict[str, Any]:
    """
    Simulate IDF and iterate through the results to extract results from CSV and/or HTML files.
    
    Args:
        idf: IDF object to simulate
        results: List of result types to extract (e.g., ['csv', 'html', 'sql'])
    
    Returns:
        Dictionary containing the extracted results
    """
    # Run the simulation
    try:
        idf.run()
        simulation_successful = True
    except Exception as e:
        simulation_successful = False
        return {
            'success': False,
            'error': str(e),
            'results': {}
        }
    
    extracted_results = {}
    
    # Get the output directory from the IDF run
    output_directory = idf.run_output_directory if hasattr(idf, 'run_output_directory') else 'run_outputs'
    
    for result_type in results:
        if result_type.lower() == 'csv':
            # Extract CSV results
            csv_results = extract_csv_results(output_directory)
            extracted_results['csv'] = csv_results
            
        elif result_type.lower() == 'html':
            # Extract HTML results
            html_results = extract_html_results(output_directory)
            extracted_results['html'] = html_results
            
        elif result_type.lower() == 'sql':
            # Extract SQL results
            sql_results = extract_sql_results(output_directory)
            extracted_results['sql'] = sql_results
            
        elif result_type.lower() == 'eso':
            # Extract ESO results
            eso_results = extract_eso_results(output_directory)
            extracted_results['eso'] = eso_results
    
    return {
        'success': simulation_successful,
        'output_directory': output_directory,
        'results': extracted_results
    }


def extract_csv_results(output_directory: str) -> Dict[str, Any]:
    """Extract results from CSV files in the output directory."""
    csv_results = {}
    
    if not os.path.exists(output_directory):
        return csv_results
    
    for filename in os.listdir(output_directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(output_directory, filename)
            try:
                # Read CSV file
                df = pd.read_csv(filepath)
                csv_results[filename] = {
                    'data': df.to_dict('records'),
                    'columns': df.columns.tolist(),
                    'shape': df.shape
                }
            except Exception as e:
                csv_results[filename] = {'error': str(e)}
    
    return csv_results


def extract_html_results(output_directory: str) -> Dict[str, Any]:
    """Extract results from HTML files in the output directory."""
    html_results = {}
    
    if not os.path.exists(output_directory):
        return html_results
    
    for filename in os.listdir(output_directory):
        if filename.endswith('.html') or filename.endswith('.htm'):
            filepath = os.path.join(output_directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                html_results[filename] = {
                    'content': html_content,
                    'size': len(html_content)
                }
            except Exception as e:
                html_results[filename] = {'error': str(e)}
    
    return html_results


def extract_sql_results(output_directory: str) -> Dict[str, Any]:
    """Extract results from SQL files in the output directory."""
    sql_results = {}
    
    if not os.path.exists(output_directory):
        return sql_results
    
    for filename in os.listdir(output_directory):
        if filename.endswith('.sql'):
            filepath = os.path.join(output_directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    sql_content = f.read()
                sql_results[filename] = {
                    'content': sql_content,
                    'size': len(sql_content)
                }
            except Exception as e:
                sql_results[filename] = {'error': str(e)}
    
    return sql_results


def extract_eso_results(output_directory: str) -> Dict[str, Any]:
    """Extract results from ESO files in the output directory."""
    eso_results = {}
    
    if not os.path.exists(output_directory):
        return eso_results
    
    for filename in os.listdir(output_directory):
        if filename.endswith('.eso'):
            filepath = os.path.join(output_directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    eso_content = f.read()
                eso_results[filename] = {
                    'content': eso_content,
                    'size': len(eso_content)
                }
            except Exception as e:
                eso_results[filename] = {'error': str(e)}
    
    return eso_results


def apply_external_window_construction(idf, construction_name: str):
    """
    Apply a construction to all external windows in the IDF.
    
    Args:
        idf: IDF object to modify
        construction_name: Name of the construction to apply
    
    Returns:
        Modified IDF object
    """
    # Find all fenestration surfaces (windows)
    if 'FENESTRATIONSURFACE:DETAILED' in idf.idfobjects:
        for window in idf.idfobjects['FENESTRATIONSURFACE:DETAILED']:
            # Check if it's an external window
            if hasattr(window, 'Surface_Type') and window.Surface_Type == 'Window':
                # Apply the construction
                window.Construction_Name = construction_name
    
    # Also check for simple fenestration surfaces
    if 'FENESTRATIONSURFACE:SIMPLE' in idf.idfobjects:
        for window in idf.idfobjects['FENESTRATIONSURFACE:SIMPLE']:
            # Apply the construction
            window.Construction_Name = construction_name
    
    return idf


def apply_roof_construction(idf, construction_name: str):
    """
    Apply a construction to all roof surfaces in the IDF.
    
    Args:
        idf: IDF object to modify
        construction_name: Name of the construction to apply
    
    Returns:
        Modified IDF object
    """
    # Find all building surfaces
    if 'BUILDINGSURFACE:DETAILED' in idf.idfobjects:
        for surface in idf.idfobjects['BUILDINGSURFACE:DETAILED']:
            # Check if it's a roof
            if hasattr(surface, 'Surface_Type') and surface.Surface_Type == 'Roof':
                # Apply the construction
                surface.Construction_Name = construction_name
    
    # Also check for simple building surfaces
    if 'BUILDINGSURFACE:SIMPLE' in idf.idfobjects:
        for surface in idf.idfobjects['BUILDINGSURFACE:SIMPLE']:
            # Check if it's a roof
            if hasattr(surface, 'Surface_Type') and surface.Surface_Type == 'Roof':
                # Apply the construction
                surface.Construction_Name = construction_name
    
    return idf


def apply_window_construction(idf, construction_name: str):
    """
    Apply a construction to all windows (both internal and external) in the IDF.
    
    Args:
        idf: IDF object to modify
        construction_name: Name of the construction to apply
    
    Returns:
        Modified IDF object
    """
    # Find all fenestration surfaces (windows)
    if 'FENESTRATIONSURFACE:DETAILED' in idf.idfobjects:
        for window in idf.idfobjects['FENESTRATIONSURFACE:DETAILED']:
            # Apply the construction to all windows
            window.Construction_Name = construction_name
    
    # Also check for simple fenestration surfaces
    if 'FENESTRATIONSURFACE:SIMPLE' in idf.idfobjects:
        for window in idf.idfobjects['FENESTRATIONSURFACE:SIMPLE']:
            # Apply the construction to all windows
            window.Construction_Name = construction_name
    
    return idf


def parse_eso_variables(
    eso_file_path: str,
    variable_patterns: List[str],
    reporting_frequency: Optional[str] = None,
    zone_include: Optional[List[str]] = None,
    zone_exclude: Optional[List[str]] = None,
    return_as: str = 'dataframe',
    timestamp_start: Optional[datetime] = None,
    timestamp_freq: str = 'H',
) -> Dict[str, pd.DataFrame]:
    """
    Parse an EnergyPlus .eso file and extract variables matching given patterns and criteria.
    Args:
        eso_file_path: Path to the .eso file
        variable_patterns: List of regex patterns to match variable names (e.g., ['Zone Mean Air Temperature'])
        reporting_frequency: Only include variables with this reporting frequency (e.g., 'Hourly'), or None for any
        zone_include: Only include zones whose name matches one of these (case-insensitive, substring match)
        zone_exclude: Exclude zones whose name matches one of these (case-insensitive, substring match)
        return_as: 'dataframe' (default) or 'array'
        timestamp_start: If timestamps are not present, generate from this start datetime
        timestamp_freq: Frequency for generated timestamps (default 'H' for hourly)
    Returns:
        Dict mapping variable keys (e.g., 'zone_SW:BR') to DataFrames or arrays with columns: timestamp, value
    """
    # Compile patterns
    compiled_patterns = [re.compile(pat, re.IGNORECASE) for pat in variable_patterns]
    if zone_include:
        zone_include = [z.lower() for z in zone_include]
    if zone_exclude:
        zone_exclude = [z.lower() for z in zone_exclude]
    
    variable_mapping = {}
    zone_ids = {}
    variable_keys = {}
    with open(eso_file_path, 'r') as file:
        lines = file.readlines()
    # First pass: find variable definitions
    for line in lines:
        line = line.strip()
        if not line or ',' not in line:
            continue
        parts = line.split(',')
        if len(parts) < 4:
            continue
        try:
            var_id = int(parts[0])
        except ValueError:
            continue
        zone_or_env = parts[2].strip()
        var_name = parts[3].strip()
        # Match variable name pattern
        if not any(pat.search(var_name) for pat in compiled_patterns):
            continue
        # Match reporting frequency if specified
        if reporting_frequency and reporting_frequency.lower() not in var_name.lower():
            continue
        # Zone inclusion/exclusion
        zone_key = zone_or_env
        if zone_include and not any(z in zone_key.lower() for z in zone_include):
            continue
        if zone_exclude and any(z in zone_key.lower() for z in zone_exclude):
            continue
        # Build key
        key = f"{zone_key}:{var_name}" if zone_key else var_name
        variable_mapping[key] = var_id
        zone_ids[var_id] = key
        variable_keys[var_id] = key
    # Second pass: extract data
    timestamps = []
    data = {key: [] for key in variable_mapping}
    current_date = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Timestamp line
        if re.match(r'^\d{4},\d{1,2},\d{1,2},\d{1,2},\d{1,2},\d{1,2}', line):
            parts = line.split(',')
            year, month, day, hour, minute, second = map(int, parts[:6])
            current_date = datetime(year, month, day, hour, minute, second)
            timestamps.append(current_date)
        # Data line
        elif re.match(r'^\d+,[-+]?\d*\.?\d+', line):
            parts = line.split(',')
            var_id = int(parts[0])
            value = float(parts[1])
            if var_id in variable_keys:
                data[variable_keys[var_id]].append(value)
    # Generate timestamps if not present
    if not timestamps:
        # Use provided start or default to Jan 1, 2021
        if timestamp_start is None:
            timestamp_start = datetime(2021, 1, 1, 0, 0, 0)
        # Assume all variables have same length
        n = max(len(vals) for vals in data.values()) if data else 0
        # Map freq to pandas unit
        freq_map = {'h': 'h', 'hour': 'h', 'd': 'd', 'day': 'd', 'm': 'm', 'min': 'm'}
        freq_key = timestamp_freq.lower() if timestamp_freq else 'h'
        unit = freq_map.get(freq_key, 'h')
        timestamps = [timestamp_start + pd.Timedelta(i, unit=unit) for i in range(n)]
    # Build result
    result = {}
    for key, values in data.items():
        df = pd.DataFrame({
            'timestamp': timestamps[:len(values)],
            'value': values
        })
        result[key] = df if return_as == 'dataframe' else (np.array(timestamps[:len(values)]), np.array(values))
    
    return result

def parse_eso_file(eso_file_path: str) -> Tuple[Dict[str, List[float]], List[datetime], Dict[str, int]]:
    """
    Parse EnergyPlus .eso file to extract zone temperatures and outdoor air temperature.
    
    Args:
        eso_file_path: Path to the .eso file
        
    Returns:
        Tuple of (temperature_data, timestamps, variable_mapping)
    """
    temperature_data = {}
    timestamps = []
    variable_mapping = {}
    
    with open(eso_file_path, 'r') as file:
        lines = file.readlines()
    
    # First pass: find variable definitions and build mapping
    for line in lines:
        line = line.strip()
        if not line or not ',' in line:
            continue
        parts = line.split(',')
        if len(parts) < 4:
            continue
        try:
            var_id = int(parts[0])
        except ValueError:
            continue  # Skip lines where first field is not an integer
        zone_or_env = parts[2].strip()
        var_name = parts[3].strip()
        # Match outdoor air temperature
        if 'Site Outdoor Air Drybulb Temperature' in var_name:
            variable_mapping['outdoor'] = var_id
        # Match all zone mean air temperature variables (hourly only)
        if 'Zone Mean Air Temperature [C]' in var_name and '!Hourly' in var_name:
            zone_key = f'zone_{zone_or_env}'
            variable_mapping[zone_key] = var_id
    
    # Second pass: extract data
    current_date = None
    
    # Track data for each variable
    for var_name in variable_mapping:
        temperature_data[var_name] = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a timestamp line (format: YYYY,MM,DD,HH,MM,SS)
        if re.match(r'^\d{4},\d{1,2},\d{1,2},\d{1,2},\d{1,2},\d{1,2}', line):
            parts = line.split(',')
            year, month, day, hour, minute, second = map(int, parts[:6])
            current_date = datetime(year, month, day, hour, minute, second)
            timestamps.append(current_date)
        # Check if this is a data line (format: VAR_ID,VALUE)
        elif re.match(r'^\d+,[-+]?\d*\.?\d+', line):
            parts = line.split(',')
            var_id = int(parts[0])
            value = float(parts[1])
            for var_name, vid in variable_mapping.items():
                if vid == var_id:
                    if var_name in temperature_data:
                        temperature_data[var_name].append(value)
                    break
    
    # Generate timestamps for variables that don't have explicit timestamp lines
    for var_name, values in temperature_data.items():
        if len(values) > 0 and len(timestamps) == 0:
            start_date = datetime(2021, 1, 1, 0, 0, 0)  # Assuming 2021 based on weather file
            generated_timestamps = []
            for i in range(len(values)):
                timestamp = start_date + timedelta(hours=i)
                generated_timestamps.append(timestamp)
            if len(timestamps) == 0:
                timestamps.extend(generated_timestamps)
            break
    
    return temperature_data, timestamps, variable_mapping


def extract_zone_temperatures(temperature_data: Dict[str, List[float]], timestamps: List[datetime]) -> Dict[str, pd.DataFrame]:
    """
    Convert temperature data and timestamps to zone temperature DataFrames.
    
    Args:
        temperature_data: Dictionary with variable names as keys and temperature values as lists
        timestamps: List of datetime objects corresponding to the temperature data
    
    Returns:
        Dictionary with zone names as keys and temperature DataFrames as values
    """
    # Convert to DataFrames
    zone_temps = {}
    
    for var_name, values in temperature_data.items():
        if var_name.startswith('zone_'):
            zone_name = var_name.replace('zone_', '')
            
            # Create DataFrame with timestamps and temperatures
            df = pd.DataFrame({
                'timestamp': timestamps[:len(values)],
                'temperature': values
            })
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Add date columns for easier filtering
            df['date'] = df['timestamp'].dt.date
            df['month'] = df['timestamp'].dt.month
            df['day'] = df['timestamp'].dt.day
            df['hour'] = df['timestamp'].dt.hour
            
            zone_temps[zone_name] = df
    
    # Also extract outdoor temperature
    if 'outdoor' in temperature_data:
        outdoor_df = pd.DataFrame({
            'timestamp': timestamps[:len(temperature_data['outdoor'])],
            'temperature': temperature_data['outdoor']
        })
        outdoor_df['timestamp'] = pd.to_datetime(outdoor_df['timestamp'])
        outdoor_df['date'] = outdoor_df['timestamp'].dt.date
        outdoor_df['month'] = outdoor_df['timestamp'].dt.month
        outdoor_df['day'] = outdoor_df['timestamp'].dt.day
        outdoor_df['hour'] = outdoor_df['timestamp'].dt.hour
        
        zone_temps['outdoor'] = outdoor_df
    
    return zone_temps

