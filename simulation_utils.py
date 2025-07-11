import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import re
import json
import shutil

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


def extract_district_heating_from_htm(htm_file_path: str) -> float:
    """
    Extract District Heating value from the End Uses table in eplustbl.htm file.
    
    Args:
        htm_file_path: Path to the eplustbl.htm file
        
    Returns:
        District Heating value in kWh as a float
        
    Raises:
        FileNotFoundError: If the HTML file doesn't exist
        ValueError: If the District Heating value cannot be found or parsed
    """
    if not os.path.exists(htm_file_path):
        raise FileNotFoundError(f"HTML file not found: {htm_file_path}")
    
    district_heating_value = None
    
    with open(htm_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Use regex to find the "Total End Uses" row and extract the District Heating value
    # The District Heating column is the 12th column (index 11)
    import re
    
    # Pattern to match the Total End Uses row and capture the 12th column value
    pattern = r'<td[^>]*>Total End Uses</td>(?:\s*<td[^>]*>[^<]*</td>){11}\s*<td[^>]*>([^<]*)</td>'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        value_str = match.group(1).strip()
        try:
            district_heating_value = float(value_str)
        except ValueError:
            raise ValueError(f"Could not parse District Heating value: '{value_str}'")
    else:
        # Alternative approach: look for the specific line with "Total End Uses"
        lines = content.split('\n')
        for line in lines:
            if 'Total End Uses' in line and '<td' in line:
                # Extract all td elements from this line
                td_matches = re.findall(r'<td[^>]*>([^<]*)</td>', line)
                if len(td_matches) >= 13:  # Need at least 13 columns
                    value_str = td_matches[12].strip()  # 12th column (index 12)
                    try:
                        district_heating_value = float(value_str)
                        break
                    except ValueError:
                        continue
    
    if district_heating_value is None:
        raise ValueError("District Heating value not found in the End Uses table")
    
    return district_heating_value


# ============================================================================
# CACHING FUNCTIONS
# ============================================================================

# Global cache for multiprocessing
CACHE_DIR = "cache"
GLOBAL_CACHE = {}

def initialize_cache():
    """Initialize the cache directory for the current run."""
    try:
        # Create cache directory if it doesn't exist
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"Initialized cache directory: {CACHE_DIR}")
    except Exception as e:
        print(f"Warning: Could not initialize cache directory: {e}")

def check_and_update_cache(combination_key):
    """Check if combination exists in cache and update local cache from files."""
    global GLOBAL_CACHE
    
    # Check local cache first
    if combination_key in GLOBAL_CACHE:
        return GLOBAL_CACHE[combination_key]
    
    # Check all cache files in the cache directory
    try:
        if os.path.exists(CACHE_DIR):
            for filename in os.listdir(CACHE_DIR):
                if filename.endswith('.json'):
                    filepath = os.path.join(CACHE_DIR, filename)
                    try:
                        with open(filepath, 'r') as f:
                            cache_data = json.load(f)
                            temp_cache = {eval(k): v for k, v in cache_data.items()}
                            if combination_key in temp_cache:
                                print(f"DUPLICATE DETECTED in {filename}! Reusing results for: {combination_key}")
                                result = temp_cache[combination_key]
                                # Update local cache
                                GLOBAL_CACHE[combination_key] = result
                                return result
                    except Exception as e:
                        # Silently continue if individual file read fails
                        pass
    except Exception as e:
        # Silently continue if cache directory access fails
        pass
    
    # Also reload cache from all files to catch new entries
    try:
        if os.path.exists(CACHE_DIR):
            for filename in os.listdir(CACHE_DIR):
                if filename.endswith('.json'):
                    filepath = os.path.join(CACHE_DIR, filename)
                    try:
                        with open(filepath, 'r') as f:
                            cache_data = json.load(f)
                            # Update local cache with any new entries from file
                            for k, v in cache_data.items():
                                key_tuple = eval(k)
                                if key_tuple not in GLOBAL_CACHE:
                                    GLOBAL_CACHE[key_tuple] = v
                    except Exception as e:
                        # Silently continue if individual file read fails
                        pass
            # Check again after reloading
            if combination_key in GLOBAL_CACHE:
                print(f"DUPLICATE DETECTED after reload! Reusing results for: {combination_key}")
                return GLOBAL_CACHE[combination_key]
    except Exception as e:
        # Silently continue if cache directory access fails
        pass
    
    return None  # Not found in cache

def save_global_cache():
    """Save the current process cache to its own file in the cache directory."""
    try:
        # Create cache directory if it doesn't exist
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Create a unique filename for this process
        process_id = os.getpid()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        cache_filename = f"cache_pid{process_id}_{timestamp}.json"
        cache_filepath = os.path.join(CACHE_DIR, cache_filename)
        
        # Convert tuple keys to strings for JSON serialization
        cache_data = {str(k): v for k, v in GLOBAL_CACHE.items()}
        
        # Save to this process's cache file
        with open(cache_filepath, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"Saved {len(GLOBAL_CACHE)} solutions to cache file {cache_filename}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

def merge_all_cache_files(keep_cache_dir=False):
    """Merge all cache files and save as final cache with timestamp."""
    try:
        # Create timestamp for final cache file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_cache_file = f"final_cache_{timestamp}.json"
        
        # Load all cache files
        merged_cache = {}
        if os.path.exists(CACHE_DIR):
            for filename in os.listdir(CACHE_DIR):
                if filename.endswith('.json'):
                    filepath = os.path.join(CACHE_DIR, filename)
                    try:
                        with open(filepath, 'r') as f:
                            cache_data = json.load(f)
                            # Convert string keys back to tuples and merge
                            for k, v in cache_data.items():
                                merged_cache[eval(k)] = v
                    except Exception as e:
                        print(f"Warning: Could not load cache file {filename}: {e}")
        
        # Save merged cache to final file
        final_cache_data = {str(k): v for k, v in merged_cache.items()}
        with open(final_cache_file, 'w') as f:
            json.dump(final_cache_data, f, indent=2)
        
        print(f"Saved final cache with {len(merged_cache)} solutions to {final_cache_file}")
        
        # Clean up cache directory (optional)
        if not keep_cache_dir:
            try:
                shutil.rmtree(CACHE_DIR)
                print(f"Cleaned up cache directory {CACHE_DIR}")
            except Exception as e:
                print(f"Warning: Could not clean up cache directory: {e}")
        else:
            print(f"Keeping cache directory {CACHE_DIR} for debugging")
            
    except Exception as e:
        print(f"Warning: Could not merge cache files: {e}")

