import pandas as pd
from typing import Dict, List
import re

# Pre-compile regex patterns for performance
TIMESTAMP_PATTERN = re.compile(r'^\d{4},\d{1,2},\d{1,2},\d{1,2},\d{1,2},\d{1,2}')
DATA_PATTERN = re.compile(r'^\d+,[-+]?\d*\.?\d+')

# Constants for ASHRAE 55 Adaptive method
ASHRAE_ALPHA = 0.31
ASHRAE_BETA = 21.3


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
    
    # Convert timestamps to datetime and extract date
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    # Calculate daily mean temperatures
    daily_means = df.groupby('date')['temperature'].mean()
    
    # Calculate 7-day running mean
    # For the first 7 days, use available data (minimum 3 days required by ASHRAE 55)
    running_mean = daily_means.rolling(window=7, min_periods=3, center=False).mean()

    # Handle NaN values for the first few days
    # ASHRAE 55 allows using available data for the first 7 days (minimum 3 days)
    # Forward fill any remaining NaN values
    running_mean = running_mean.ffill()

    # If there are still NaN values at the beginning, backfill them
    running_mean = running_mean.bfill()
    
    # Map daily running means back to hourly data using merge
    # Convert running_mean to DataFrame for merging
    running_mean_df = running_mean.reset_index()
    running_mean_df.columns = ['date', 'running_mean']
    
    # Merge with original data
    df = df.merge(running_mean_df, on='date', how='left')
    
    # ASHRAE 55 Adaptive 80% criteria
    # T_adaptive = 0.31 * T_running_mean + 17.8
    threshold_temps = ASHRAE_ALPHA * df['running_mean'] + ASHRAE_BETA
    
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
    
    zone_temp_series = zone_temps['temperature'].iloc[:min_length].reset_index(drop=True)
    threshold_series = threshold_temps.iloc[:min_length].reset_index(drop=True)
    
    # Count hours where zone temperature exceeds threshold
    overheating_hours = (zone_temp_series > threshold_series).sum()
    
    return overheating_hours


def analyze_overheating(temp_data: Dict[str, pd.DataFrame], 
                       summer_months: List[int] | None = None) -> dict:
    """
    Perform ASHRAE 55 Adaptive 80% overheating analysis on temperature data.
    
    Args:
        temp_data: Dictionary with zone names as keys and temperature DataFrames as values
        summer_months: List of months to consider for summer analysis (default: May-Sep)
        
    Returns:
        Dictionary with analysis results
    """
    if summer_months is None:
        summer_months = [5, 6, 7, 8, 9]  # May to September
    
    if not temp_data:
        raise ValueError("No temperature data provided")
    
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
    threshold_temps = calculate_ashrae_adaptive_threshold(
        pd.Series(summer_outdoor['temperature']), 
        pd.Series(summer_outdoor['timestamp'])
    )
    
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
        overheating_hours = calculate_overheating_hours(pd.DataFrame(summer_zone), threshold_temps)
        
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
    if 'excluded_zones' in results:
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
