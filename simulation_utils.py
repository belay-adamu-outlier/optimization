import os
import csv
import json
from typing import Dict, List, Tuple, Any, Generator, Optional
from eppy.modeleditor import IDF
import pandas as pd


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
