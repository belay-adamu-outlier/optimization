import os
import pandas as pd
from eppy.modeleditor import IDF
from eppy.runner.run_functions import runIDFs
import tempfile
import shutil


class EnergyPlusRunner:
    """
    Utility class for running EnergyPlus simulations using eppy.
    Handles simulation execution and result extraction.
    """
    
    def __init__(self, idd_file_path):
        """
        Initialize the EnergyPlus runner.
        
        :param idd_file_path: Path to the EnergyPlus IDD file
        """
        IDF.setiddname(idd_file_path)
        self.idd_file_path = idd_file_path
        self.simulation_results = None
        self.output_directory = None
        self.csv_data = None
    
    def run_simulation(self, idf_file_path, weather_file_path, output_directory=None, 
                      **run_options):
        """
        Run an EnergyPlus simulation and store results for later querying.
        
        :param idf_file_path: Path to the IDF file
        :param weather_file_path: Path to the weather file (.epw)
        :param output_directory: Directory for output files (optional)
        :param run_options: Additional options for EnergyPlus run
        :return: Dictionary containing simulation results
        """
        # Create temporary output directory if none specified
        if output_directory is None:
            output_directory = tempfile.mkdtemp(prefix='eplus_run_')
        
        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        # Store output directory for later use
        self.output_directory = output_directory
        
        # Load IDF file
        idf = IDF(idf_file_path, weather_file_path)
        
        # Set up default run options
        default_options = {
            'output_directory': output_directory,
            'readvars': True,  # Generate CSV output
            'expandobjects': True,
            'output_prefix': os.path.splitext(os.path.basename(idf_file_path))[0],
            'output_suffix': 'C'
        }
        
        # Update with user-provided options
        default_options.update(run_options)
        
        try:
            # Run the simulation
            print(f"Running EnergyPlus simulation...")
            print(f"IDF: {idf_file_path}")
            print(f"Weather: {weather_file_path}")
            print(f"Output directory: {output_directory}")
            
            idf.run(**default_options)
            
            # Load and store CSV data
            self._load_csv_data()
            
            # Store basic simulation info
            self.simulation_results = {
                'idf_file': idf_file_path,
                'weather_file': weather_file_path,
                'output_directory': output_directory,
                'summary': {
                    'total_records': len(self.csv_data) if self.csv_data is not None else 0,
                    'variables_available': list(self.csv_data.columns) if self.csv_data is not None else [],
                    'output_file': self._get_csv_file_path()
                }
            }
            
            print(f"Simulation completed! Results stored for querying.")
            return self.simulation_results
            
        except Exception as e:
            print(f"Error running EnergyPlus simulation: {e}")
            raise
    
    def _load_csv_data(self):
        """Load CSV data from the output directory."""
        csv_file = self._get_csv_file_path()
        if csv_file and os.path.exists(csv_file):
            try:
                self.csv_data = pd.read_csv(csv_file)
                print(f"Loaded CSV data with {len(self.csv_data)} records and {len(self.csv_data.columns)} columns")
            except Exception as e:
                print(f"Error loading CSV data: {e}")
                self.csv_data = None
        else:
            print("Warning: No CSV output file found")
            self.csv_data = None
    
    def _get_csv_file_path(self):
        """Get the path to the CSV output file."""
        if not self.output_directory:
            return None
        
        csv_files = [f for f in os.listdir(self.output_directory) if f.endswith('.csv')]
        if csv_files:
            return os.path.join(self.output_directory, csv_files[0])
        return None
    
    def get_results(self, output_variables=None):
        """
        Get results from the stored simulation data.
        
        :param output_variables: List of specific variables to extract (optional)
        :return: Dictionary of results
        """
        if self.csv_data is None:
            raise ValueError("No simulation results available. Run simulation first.")
        
        results = {}
        
        # Default variables to extract if none specified
        if output_variables is None:
            output_variables = [
                'Zone Mean Air Temperature',
                'District:Heating'
            ]
        
        # Extract specified variables
        for variable in output_variables:
            # Find columns that match the variable name
            matching_cols = [col for col in self.csv_data.columns if variable.lower() in col.lower()]
            
            if matching_cols:
                for col in matching_cols:
                    results[col] = self.csv_data[col].tolist()
            else:
                print(f"Warning: Variable '{variable}' not found in output")
        
        # Add timestamp information if available
        if 'Date/Time' in self.csv_data.columns:
            results['timestamp'] = self.csv_data['Date/Time'].tolist()
        
        return results
    
    def get_zone_temperatures(self):
        """
        Get hourly zone temperatures from stored simulation results.
        
        :return: Dictionary with zone temperature data
        """
        return self.get_results(['Zone Mean Air Temperature'])
    
    def get_district_heating(self):
        """
        Get district heating data from stored simulation results.
        
        :return: Dictionary with district heating data
        """
        return self.get_results(['District:Heating'])
    
    def get_available_variables(self):
        """
        Get list of all available output variables from the simulation.
        
        :return: List of available variable names
        """
        if self.csv_data is None:
            return []
        return list(self.csv_data.columns)
    
    def get_summary(self):
        """
        Get summary information about the simulation results.
        
        :return: Dictionary with summary information
        """
        if self.simulation_results is None:
            return None
        return self.simulation_results['summary']
    
    def cleanup(self):
        """Clean up temporary output directory."""
        if self.output_directory and self.output_directory.startswith(tempfile.gettempdir()):
            try:
                shutil.rmtree(self.output_directory)
                print(f"Cleaned up temporary directory: {self.output_directory}")
            except Exception as e:
                print(f"Error cleaning up temporary directory: {e}")


def run_energyplus_simulation(idf_file_path, weather_file_path, idd_file_path, 
                             output_variables=None, output_directory=None, **run_options):
    """
    Convenience function to run a single EnergyPlus simulation and return results.
    
    :param idf_file_path: Path to the IDF file
    :param weather_file_path: Path to the weather file
    :param idd_file_path: Path to the EnergyPlus IDD file
    :param output_variables: List of output variables to extract
    :param output_directory: Directory for output files
    :param run_options: Additional EnergyPlus run options
    :return: Dictionary containing simulation results
    """
    runner = EnergyPlusRunner(idd_file_path)
    runner.run_simulation(idf_file_path, weather_file_path, output_directory=output_directory, **run_options)
    
    if output_variables:
        return runner.get_results(output_variables)
    else:
        return runner.get_results()


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    idd_file = 'C:/EnergyPlusV9-4-0/Energy+.idd'
    idf_file = './Building.idf'
    weather_file = './Vancouver - 2021 AMY.epw'
    
    # Create runner
    runner = EnergyPlusRunner(idd_file)
    
    # Run simulation once
    print("Running simulation...")
    runner.run_simulation(idf_file, weather_file)
    
    # Now query different results without re-running simulation
    print("\nQuerying results from stored simulation data:")
    
    # Get available variables
    available_vars = runner.get_available_variables()
    print(f"Available variables: {available_vars}")
    
    # Get summary
    summary = runner.get_summary()
    print(f"Summary: {summary}")
    
    # Get zone temperatures
    zone_results = runner.get_zone_temperatures()
    print(f"Zone temperature data keys: {list(zone_results.keys())}")
    
    # Get district heating
    heating_results = runner.get_district_heating()
    print(f"District heating data keys: {list(heating_results.keys())}")
    
    # Get custom variables
    custom_results = runner.get_results(['Zone Air Relative Humidity'])
    print(f"Custom results keys: {list(custom_results.keys())}")
    
    # Clean up
    runner.cleanup() 