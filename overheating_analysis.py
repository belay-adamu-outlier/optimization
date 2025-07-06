import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os
import re

# Pre-compile regex patterns for performance
TIMESTAMP_PATTERN = re.compile(r'^\d{4},\d{1,2},\d{1,2},\d{1,2},\d{1,2},\d{1,2}')
DATA_PATTERN = re.compile(r'^\d+,[-+]?\d*\.?\d+')

# Constants for ASHRAE 55 Adaptive method
ASHRAE_ALPHA = 0.31
ASHRAE_BETA = 17.8

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
            # Exclude zones with names containing Corridor, Elevator, Stairs (case-insensitive)
            if any(excl.lower() in zone_or_env.lower() for excl in ['corridor', 'elevator', 'stairs']):
                continue
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


def extract_zone_temperatures(output_directory: str, excluded_zones: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Extract zone mean air temperatures from EnergyPlus .eso output file.
    
    Args:
        output_directory: Directory containing EnergyPlus output files
        excluded_zones: List of zone names to exclude from analysis
    
    Returns:
        Dictionary with zone names as keys and temperature DataFrames as values
    """
    if excluded_zones is None:
        excluded_zones = ['Corridor', 'Elevator', 'Stairs']
    
    # Look for .eso file
    eso_file = os.path.join(output_directory, 'eplusout.eso')
    
    if not os.path.exists(eso_file):
        raise FileNotFoundError(f"EnergyPlus .eso file not found: {eso_file}")
    
    # Parse the .eso file
    temperature_data, timestamps, variable_mapping = parse_eso_file(eso_file)
    
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


def calculate_ashrae_adaptive_threshold(outdoor_temps: pd.Series, timestamps: pd.Series) -> pd.Series:
    """
    Calculate ASHRAE 55 Adaptive 80% threshold temperatures using 7-day running mean.
    
    Args:
        outdoor_temps: Series of outdoor air temperatures
        timestamps: Series of timestamps corresponding to outdoor temperatures
        
    Returns:
        Series of adaptive threshold temperatures
    """
    # Create DataFrame with timestamps and temperatures
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': outdoor_temps
    })
    
    # Calculate daily mean temperatures
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_means = df.groupby('date')['temperature'].mean()
    
    # Calculate 7-day running mean
    # For the first 7 days, use available data (minimum 3 days required by ASHRAE 55)
    running_mean = daily_means.rolling(window=7, min_periods=3, center=False).mean()
    
    # Forward fill any NaN values (for days with insufficient data)
    running_mean = running_mean.ffill()
    
    # Map daily running means back to hourly data
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    # Convert running_mean to a dictionary for mapping
    running_mean_dict = running_mean.to_dict()
    df['running_mean'] = df['date'].map(running_mean_dict)
    
    # ASHRAE 55 Adaptive 80% criteria
    # T_adaptive = 0.31 * T_running_mean + 17.8
    threshold_temps = 0.31 * df['running_mean'] + 17.8
    
    return threshold_temps


def calculate_overheating_hours(zone_temps: pd.DataFrame, threshold_temps: pd.Series) -> int:
    """
    Calculate overheating hours for a zone.
    
    Args:
        zone_temps: DataFrame with zone temperature data
        threshold_temps: Series with adaptive threshold temperatures
        
    Returns:
        Number of overheating hours
    """
    # Ensure we have the same number of data points
    min_length = min(len(zone_temps), len(threshold_temps))
    
    zone_temp_series = zone_temps['temperature'].iloc[:min_length]
    threshold_series = threshold_temps.iloc[:min_length]
    
    # Count hours where zone temperature exceeds threshold
    overheating_hours = (zone_temp_series > threshold_series).sum()
    
    return overheating_hours


def perform_overheating_analysis(output_directory: str, 
                                excluded_zones: List[str] = None,
                                summer_months: List[int] = None) -> dict:
    """
    Perform ASHRAE 55 Adaptive 80% overheating analysis.
    
    Args:
        output_directory: Directory containing EnergyPlus output files
        excluded_zones: List of zone names to exclude from analysis
        summer_months: List of months to consider for summer analysis (default: May-Sep)
        
    Returns:
        Dictionary with analysis results
    """
    if excluded_zones is None:
        excluded_zones = ['Corridor', 'Elevator', 'Stairs']
    
    if summer_months is None:
        summer_months = [5, 6, 7, 8, 9]  # May to September
    
    print(f"Extracting temperature data from {output_directory}...")
    
    # Extract temperature data
    temp_data = extract_zone_temperatures(output_directory, excluded_zones)
    
    if not temp_data:
        raise ValueError("No temperature data found")
    
    # Get outdoor temperature data
    if 'outdoor' not in temp_data:
        raise ValueError("Outdoor temperature data not found")
    
    outdoor_data = temp_data['outdoor']
    
    # Filter for summer months
    summer_outdoor = outdoor_data[outdoor_data['month'].isin(summer_months)]
    
    if len(summer_outdoor) == 0:
        raise ValueError(f"No data found for summer months {summer_months}")
    
    print(f"Found {len(summer_outdoor)} hours of summer data")
    
    # Calculate adaptive thresholds
    threshold_temps = calculate_ashrae_adaptive_threshold(summer_outdoor['temperature'], summer_outdoor['timestamp'])
    
    # Calculate overheating for each zone
    zone_results = {}
    total_overheating_hours = 0
    zone_count = 0
    
    for zone_name, zone_data in temp_data.items():
        if zone_name == 'outdoor':
            continue
            
        # Filter zone data for summer months
        summer_zone = zone_data[zone_data['month'].isin(summer_months)]
        
        if len(summer_zone) == 0:
            continue
        
        # Calculate overheating hours
        overheating_hours = calculate_overheating_hours(summer_zone, threshold_temps)
        
        zone_results[zone_name] = {
            'overheating_hours': overheating_hours,
            'total_hours': len(summer_zone),
            'overheating_percentage': (overheating_hours / len(summer_zone)) * 100 if len(summer_zone) > 0 else 0
        }
        
        total_overheating_hours += overheating_hours
        zone_count += 1
    
    # Calculate average overheating hours
    average_overheating_hours = total_overheating_hours / zone_count if zone_count > 0 else 0
    
    results = {
        'zone_results': zone_results,
        'total_overheating_hours': total_overheating_hours,
        'zone_count': zone_count,
        'average_overheating_hours': average_overheating_hours,
        'summer_months': summer_months,
        'excluded_zones': excluded_zones,
        'analysis_period': f"{len(summer_outdoor)} hours ({summer_months[0]}-{summer_months[-1]})"
    }
    
    return results


def print_overheating_analysis_results(results: dict) -> None:
    """
    Print overheating analysis results in a formatted way.
    
    Args:
        results: Results dictionary from perform_overheating_analysis
    """
    print("\n" + "="*60)
    print("ASHRAE 55 ADAPTIVE 80% OVERHEATING ANALYSIS")
    print("="*60)
    
    print(f"\nAnalysis Period: {results['analysis_period']}")
    print(f"Summer Months: {results['summer_months']}")
    print(f"Excluded Zones: {results['excluded_zones']}")
    print(f"Total Zones Analyzed: {results['zone_count']}")
    
    print(f"\nOVERALL RESULTS:")
    print(f"  Total Overheating Hours: {results['total_overheating_hours']:.1f}")
    print(f"  Average Overheating Hours: {results['average_overheating_hours']:.1f}")
    
    print(f"\nZONE-BY-ZONE RESULTS:")
    print("-" * 80)
    print(f"{'Zone Name':<20} {'Overheating Hours':<18} {'Total Hours':<12} {'Percentage':<12}")
    print("-" * 80)
    
    for zone_name, zone_data in results['zone_results'].items():
        print(f"{zone_name:<20} {zone_data['overheating_hours']:<18.1f} "
              f"{zone_data['total_hours']:<12} {zone_data['overheating_percentage']:<12.1f}%")
    
    print("-" * 80)
    print(f"\nAnalysis completed successfully!")


def perform_overheating_analysis_optimized(output_directory: str, 
                                         excluded_zones: List[str] = None,
                                         summer_months: List[int] = None) -> dict:
    """
    Optimized version of ASHRAE 55 Adaptive 80% overheating analysis for high-frequency calls.
    
    Args:
        output_directory: Directory containing EnergyPlus output files
        excluded_zones: List of zone names to exclude from analysis
        summer_months: List of months to consider for summer analysis (default: May-Sep)
        
    Returns:
        Dictionary with analysis results (minimal data for optimization)
    """
    if excluded_zones is None:
        excluded_zones = ['Corridor', 'Elevator', 'Stairs']
    
    if summer_months is None:
        summer_months = [5, 6, 7, 8, 9]  # May to September
    
    # Extract temperature data using the working function
    temp_data = extract_zone_temperatures(output_directory, excluded_zones)
    
    if not temp_data:
        raise ValueError("No temperature data found")
    
    # Get outdoor temperature data
    if 'outdoor' not in temp_data:
        raise ValueError("Outdoor temperature data not found")
    
    outdoor_data = temp_data['outdoor']
    
    # Filter for summer months using numpy operations for speed
    summer_mask = outdoor_data['month'].isin(summer_months)
    summer_outdoor = outdoor_data[summer_mask]
    
    if len(summer_outdoor) == 0:
        raise ValueError(f"No data found for summer months {summer_months}")
    
    # Calculate adaptive thresholds
    threshold_temps = calculate_ashrae_adaptive_threshold(summer_outdoor['temperature'], summer_outdoor['timestamp'])
    
    # Calculate overheating for each zone using numpy operations
    total_overheating_hours = 0
    zone_count = 0
    
    for zone_name, zone_data in temp_data.items():
        if zone_name == 'outdoor':
            continue
            
        # Filter zone data for summer months
        summer_zone = zone_data[summer_mask]
        
        if len(summer_zone) == 0:
            continue
        
        # Calculate overheating hours using numpy for speed
        min_length = min(len(summer_zone), len(threshold_temps))
        zone_temp_series = summer_zone['temperature'].iloc[:min_length].values
        threshold_series = threshold_temps.iloc[:min_length].values
        
        # Count hours where zone temperature exceeds threshold using numpy
        overheating_hours = np.sum(zone_temp_series > threshold_series)
        
        total_overheating_hours += overheating_hours
        zone_count += 1
    
    # Calculate average overheating hours
    average_overheating_hours = total_overheating_hours / zone_count if zone_count > 0 else 0
    
    # Return minimal results for optimization
    return {
        'average_overheating_hours': average_overheating_hours,
        'total_overheating_hours': total_overheating_hours,
        'zone_count': zone_count
    } 