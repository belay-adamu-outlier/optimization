#!/usr/bin/env python3
"""
Bare minimum EnergyPlus runner using eppy.
Runs an IDF file with a weather file and returns success/failure status.
"""

import os
from eppy.modeleditor import IDF


def run_energyplus_simulation(idf_path, weather_path, idd_path=None, output_dir=None):
    """
    Run EnergyPlus simulation using eppy.
    
    Args:
        idf_path (str): Path to the IDF file
        weather_path (str): Path to the weather file (.epw)
        idd_path (str, optional): Path to the IDD file. If None, eppy will try to find it automatically.
        output_dir (str, optional): Directory for output files. If None, uses IDF directory.
    
    Returns:
        bool: True if simulation completed successfully, False otherwise
    """
    try:
        # Set up output directory
        if output_dir is None:
            output_dir = os.path.dirname(idf_path)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set the IDD file if provided
        if idd_path:
            IDF.setiddname(idd_path)
        
        # Initialize IDF with both IDF file and weather file
        idf = IDF(idf_path, weather_path)
        
        # Run simulation (no parameters needed)
        idf.run()
        
        print(f"EnergyPlus simulation completed successfully.")
        # Get output directory from IDF if available, otherwise use the provided directory
        actual_output_dir = getattr(idf, 'run_output_directory', output_dir)
        print(f"Output directory: {actual_output_dir}")
        return True
        
    except Exception as e:
        print(f"Error running EnergyPlus simulation: {e}")
        return False


# Example usage with files in current directory
if __name__ == "__main__":
    idf_file = "Building.idf"
    weather_file = "Vancouver - 2021 AMY.epw"
    idd_file = "C:/EnergyPlusV9-4-0/Energy+.idd"
    success = run_energyplus_simulation(idf_file, weather_file, idd_file)
    
    if success:
        print("Simulation completed successfully!")
    else:
        print("Simulation failed!") 