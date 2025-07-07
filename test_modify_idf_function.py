#!/usr/bin/env python3
"""
Simple test script to modify Building.idf with specific parameters and save the result.
"""

import os
import shutil
from eppy.modeleditor import IDF
from energyplus_utils import modify_idf


def test_modify_idf_function():
    """Apply modifications to Building.idf and save the result."""
    
    # Source and destination files
    source_idf = "Building.idf"
    test_idf = "Building_test_modified.idf"
    weather_file = "Vancouver - 2021 AMY.epw"
    
    # Make a copy of the original file
    shutil.copy2(source_idf, test_idf)
    
    # Set the IDD file
    idd_file = "C:/EnergyPlusV9-4-0/Energy+.idd"
    IDF.setiddname(idd_file)
    
    # Define the envelope parameters to test
    envelope_parameters = {
        'external_wall': 'R3 - Wall',
        'roof': 'R3 - Roof', 
        'window': 'Dbl LoE Elec Ref Colored 6mm/13m Arg - 1001',
        'infiltration_ach': 2.5,  # 2.5 Air Changes per Hour at 50 Pa
        'wwr_east': 0.20,    # 20%
        'wwr_west': 0.30,    # 30%
        'wwr_north': 0.25,   # 25%
        'wwr_south': 0.10    # 10%
    }
    
    # Load the IDF file
    idf = IDF(test_idf, weather_file)
    
    # Apply modifications using modify_idf function
    idf = modify_idf(idf, envelope_parameters, save=True, save_path=test_idf)
    
    # Run EnergyPlus simulation to test if the modified file works
    print("Running EnergyPlus simulation on modified IDF...")
    try:
        # Use quiet mode to suppress EnergyPlus run time logging
        idf.run(output_directory="test_simulation_output", verbose='q')
        print("✅ EnergyPlus simulation completed successfully!")
        print("No severe errors found in the modified IDF file.")
    except Exception as e:
        print(f"❌ EnergyPlus simulation failed: {e}")
        print("Check the error log for details.")
    
    # Clean up
    del idf


if __name__ == "__main__":
    test_modify_idf_function() 