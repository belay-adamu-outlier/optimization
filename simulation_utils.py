import os
import csv
from datetime import datetime
from parallel_task_runner import ParallelTaskRunner
from eplus_runner import EnergyPlusRunner


def run_simulation_task(scenario):
    """
    Run a single EnergyPlus simulation based on the provided scenario.
    This function is designed to work with ParallelTaskRunner.
    
    :param scenario: Dictionary containing simulation parameters
                    (idf_file, weather_file, idd_file)
    :return: Dictionary with simulation results
    """
    try:
        # Extract parameters from scenario
        idf_file = scenario['idf_file']
        weather_file = scenario['weather_file']
        idd_file = scenario['idd_file']
        
        # Create runner instance
        runner = EnergyPlusRunner(idd_file)
        
        # Run simulation
        runner.run_simulation(idf_file, weather_file)
        
        # Extract results
        zone_temps = runner.get_zone_temperatures()
        district_heating = runner.get_district_heating()
        
        # Get summary info
        summary = runner.get_summary()
        
        # Clean up
        runner.cleanup()
        
        # Return results with metadata
        return {
            'idf_file': idf_file,
            'weather_file': weather_file,
            'simulation_success': True,
            'total_records': summary['total_records'] if summary else 0,
            'zone_temperature_columns': list(zone_temps.keys()),
            'district_heating_columns': list(district_heating.keys()),
            'zone_temperatures': zone_temps,
            'district_heating': district_heating,
            'timestamp': zone_temps.get('timestamp', []),
            'error_message': None
        }
        
    except Exception as e:
        # Return error information
        return {
            'idf_file': scenario.get('idf_file', 'unknown'),
            'weather_file': scenario.get('weather_file', 'unknown'),
            'simulation_success': False,
            'total_records': 0,
            'zone_temperature_columns': [],
            'district_heating_columns': [],
            'zone_temperatures': {},
            'district_heating': {},
            'timestamp': [],
            'error_message': str(e)
        }


def create_simulation_tasks(idf_files, weather_file, idd_file):
    """
    Create a list of simulation task pairs for parallel execution.
    
    :param idf_files: List of IDF file paths
    :param weather_file: Path to weather file
    :param idd_file: Path to IDD file
    :return: List of (function, scenario) tuples for ParallelTaskRunner
    """
    tasks = []
    for idf_file in idf_files:
        # Create scenario dictionary with simulation parameters
        scenario = {
            'idf_file': idf_file,
            'weather_file': weather_file,
            'idd_file': idd_file
        }
        # Pair the simulation function with its scenario
        tasks.append((run_simulation_task, scenario))
    return tasks


def save_results_to_csv(results, output_csv_file):
    """
    Save simulation results to CSV file.
    
    :param results: List of simulation result dictionaries
    :param output_csv_file: Path to output CSV file
    """
    if not results:
        print("No results to save")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
    
    # Write results to CSV
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        # Define fieldnames for the CSV
        fieldnames = [
            'idf_file', 'weather_file', 'simulation_success', 'total_records',
            'zone_temperature_columns', 'district_heating_columns', 'error_message'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # Write summary row
            summary_row = {
                'idf_file': result['idf_file'],
                'weather_file': result['weather_file'],
                'simulation_success': result['simulation_success'],
                'total_records': result['total_records'],
                'zone_temperature_columns': ', '.join(result['zone_temperature_columns']),
                'district_heating_columns': ', '.join(result['district_heating_columns']),
                'error_message': result['error_message']
            }
            writer.writerow(summary_row)
    
    print(f"Results saved to: {output_csv_file}")


def create_detailed_results_csv(results, output_csv_file):
    """
    Create a detailed CSV with all time series data.
    
    :param results: List of simulation result dictionaries
    :param output_csv_file: Path to output CSV file
    """
    if not results:
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
    
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = None
        
        for result in results:
            if not result['simulation_success'] or not result['timestamp']:
                continue
            
            # Get all data columns
            all_data = {}
            all_data.update(result['zone_temperatures'])
            all_data.update(result['district_heating'])
            
            if not all_data:
                continue
            
            # Create fieldnames for this result
            if writer is None:
                fieldnames = ['idf_file', 'timestamp'] + list(all_data.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            
            # Write data rows
            for i, timestamp in enumerate(result['timestamp']):
                row = {'idf_file': result['idf_file'], 'timestamp': timestamp}
                for col, data in all_data.items():
                    if col != 'timestamp' and i < len(data):
                        row[col] = data[i]
                writer.writerow(row)
    
    print(f"Detailed results saved to: {output_csv_file}")


def print_simulation_summary(results):
    """
    Print a summary of simulation results.
    
    :param results: List of simulation result dictionaries
    """
    if not results:
        print("No results to summarize")
        return
    
    successful_sims = sum(1 for r in results if r['simulation_success'])
    failed_sims = len(results) - successful_sims
    
    print(f"\nSimulation Summary:")
    print(f"Total simulations: {len(results)}")
    print(f"Successful: {successful_sims}")
    print(f"Failed: {failed_sims}")
    
    if failed_sims > 0:
        print(f"\nFailed simulations:")
        for result in results:
            if not result['simulation_success']:
                print(f"  {os.path.basename(result['idf_file'])}: {result['error_message']}") 