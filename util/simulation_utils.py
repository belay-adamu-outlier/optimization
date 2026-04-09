import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import re
import json
from multiprocessing import Manager
from constants.shading_variables import (
    SHADING_VARIABLE_KEYS,
    SHADING_VARIABLE_BOUNDS,
    SHADING_PENALTY_WEIGHTS,
    SHADING_PENALTY_CONSTRAINTS,
)
from constants.constraints import (
    LR_OVERHEATING_MAX_HRS,
    BR_OVERHEATING_MAX_HRS,
    TEDI_MAX_VALUE,
)


def compute_shading_penalty(
    params: dict, 
    optimization_variables: dict, 
    optimization_type: str
) -> float:
    """Compute a soft penalty for shading variables using an objective penalty.

    Implements f0(x) = sum_j lambda_j * Q( g_j(x) ) where
    g_j(x) = f_j(x) - M with constraint 0 (feasible at 0). For our case we
    penalize positive magnitude of selected shading variables using
    Q(t) = max{t, 0}^2. This produces a smooth, increasing penalty when the
    variable value exceeds the 0 constraint and is 0 when at the constraint.

    Args:
        params: Candidate parameter dict.
        optimization_variables: Bounds dict (used for normalization fallback).
        optimization_type: Only 'shading' triggers penalties.

    Returns:
        float: Penalty to add to TEDI.
    """
    try:
        if optimization_type not in ('shading', 'shading_wwr') or not isinstance(params, dict):
            return 0.0

        penalty_sum = 0.0
        for key in SHADING_VARIABLE_KEYS:
            if key not in params:
                continue
            value = float(params.get(key, 0.0))
            # Per-variable constraint M (default 0.0)
            baseline = float(SHADING_PENALTY_CONSTRAINTS.get(key, 0.0))
            if key in ('south_overhang_angle', 'west_overhang_angle', 'east_overhang_angle'):
                t = max(baseline - value, 0.0)
            else:
                t = max(value - baseline, 0.0)
            bounds = SHADING_VARIABLE_BOUNDS.get(key) or {}
            max_v = bounds.get('max')
            min_v = bounds.get('min', 0.0)
            span = float(max_v - min_v) if max_v is not None else 0.0
            if span <= 0.0:
                continue
            q_t = (t / span) * (t / span)
            weight = float(SHADING_PENALTY_WEIGHTS.get(key, 1.0))
            penalty_sum += weight * q_t
        return float(penalty_sum)
    except Exception:
        return 0.0

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
        if not line or ',' not in line:
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
# SHARED CACHE FUNCTIONS
# ============================================================================
def create_shared_cache():
    """Create and return a multiprocessing.Manager dict for shared cache."""
    manager = Manager()
    return manager.dict()

def make_cache_key(params):
    """Create a string cache key from a params dict (tuple of values, stringified)."""
    return str(tuple(params.values()))

def set_in_cache(
    cache, 
    key, 
    value,
    max_overheating_hours,
    generation=None, 
    iteration=None, 
    params=None,
    max_living_room_overheating=None,
    max_bedroom_overheating=None,
    total_shading_area=None,
    feasible=None,
    **extra
):

    """Set a value in the shared cache with optional generation info and parameters."""
    if generation is not None or params is not None:
        # Feasibility should be computed by the caller (problem), default True
        if feasible is None:
            feasible = True
        cache_entry = {
            'objectives': value,
            'max_overheating_hours': max_overheating_hours,
            'generation': generation,
            'iteration': iteration,
            'params': params,
            'max_living_room_overheating': max_living_room_overheating or 0.0,
            'max_bedroom_overheating': max_bedroom_overheating or 0.0,
            'total_shading_area': total_shading_area or 0.0,
            'feasible': bool(feasible) if feasible is not None else True,
        }
        if extra:
            cache_entry.update(extra)
        cache[key] = cache_entry
    else:
        cache[key] = value

def get_from_cache(cache, key):
    """Get a value from the shared cache, or None if not present."""
    cached_value = cache.get(key, None)
    if cached_value is None:
        return None
    
    # Handle both old format (just objectives) and new format (dict with objectives and generation)
    if isinstance(cached_value, dict) and 'objectives' in cached_value:
        return cached_value['objectives'], cached_value['max_overheating_hours']
    else:
        # For simple value storage (no generation/params), return the value and -1 for max_overheating
        return cached_value, -1

def get_cache_entry(cache, key):
    """Get the full cache entry (objectives and generation info) from the shared cache."""
    return cache.get(key, None)

def get_generation_from_cache(cache, key):
    """Get the generation number for a cached solution, or None if not found."""
    cached_value = cache.get(key, None)
    if cached_value is None:
        return None
    
    # Handle both old format (just objectives) and new format (dict with objectives and generation)
    if isinstance(cached_value, dict) and 'generation' in cached_value:
        return cached_value['generation']
    else:
        return None

def save_cache_to_file(cache, generation, debug_dir="debug_cache"):
    """Save the current shared cache to a file for debugging purposes.
    Updates an existing cache file with new entries rather than creating a new file each generation.
    
    Args:
        cache: The shared cache dictionary
        generation: Current generation number
        debug_dir: Directory to save debug cache files
    """
    try:
        # Create debug directory if it doesn't exist
        os.makedirs(debug_dir, exist_ok=True)
        
        # Use a fixed filename for the cache file (no timestamp in filename)
        cache_filename = "cache_debug.json"
        cache_filepath = os.path.join(debug_dir, cache_filename)
        
        # Load existing cache data if file exists
        existing_cache = {}
        if os.path.exists(cache_filepath):
            try:
                with open(cache_filepath, 'r') as f:
                    existing_cache = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read existing cache file: {e}")
                existing_cache = {}
        
        # Convert shared cache to regular dict for JSON serialization
        current_cache = dict(cache)
        
        # Update existing cache with new entries
        # Only add entries that are not already in the existing cache
        new_entries = 0
        for key, value in current_cache.items():
            if key not in existing_cache:
                existing_cache[key] = value
                new_entries += 1
        
        # Save updated cache to file
        with open(cache_filepath, 'w') as f:
            json.dump(existing_cache, f, indent=2)
        
        print(f"Debug: Updated cache file with {new_entries} new entries (total: {len(existing_cache)} combinations)")
        
    except Exception as e:
        print(f"Warning: Could not save debug cache file: {e}")

def populate_with_cache(
    shared_cache,
    combination_key,
    is_constrained_analysis,
    out,
    optimization_type=None,
    optimization_variables=None
):
    """Try to serve results from cache and set outputs, including penalty.

    - Leaves cache contents as raw (unpenalized) so plotting/CSV remain unaffected.
    - Applies penalty on-the-fly to TEDI when setting out["F"].
    """
    if shared_cache is None:
        return False

    cached_entry = get_cache_entry(shared_cache, combination_key)
    if not (isinstance(cached_entry, dict) and 'objectives' in cached_entry):
        return False

    objectives = list(cached_entry.get('objectives') or [])
    if not objectives:
        return False

    tedi_raw = float(objectives[0])
    overheating = float(objectives[1]) if len(objectives) > 1 else 0.0

    # Compute penalty from cached params
    params = cached_entry.get('params') or {}
    penalty = compute_shading_penalty(
        params,
        optimization_variables or {},
        optimization_type or ''
    )
    is_shading = (optimization_type or '').lower() in ('shading', 'shading_wwr')
    tedi = penalty if is_shading else tedi_raw

    # For shading_wwr, return 3 objectives: [TEDI_pen, -WWR_south, -WWR_west]
    if (optimization_type or '').lower() == 'shading_wwr':
        wwr_s = float(params.get('wwr_south') or 0.0)
        wwr_w = float(params.get('wwr_west') or 0.0)
        out["F"] = [tedi, -wwr_s, -wwr_w]
    else:
        out["F"] = [tedi] if is_constrained_analysis else [tedi, overheating]

    # Constraints must match the declared constrained problem shape.
    lr = cached_entry.get('max_living_room_overheating', 0.0) or 0.0
    br = cached_entry.get('max_bedroom_overheating', 0.0) or 0.0
    if is_constrained_analysis:
        out["G"] = [
            lr - LR_OVERHEATING_MAX_HRS,
            br - BR_OVERHEATING_MAX_HRS,
            tedi - TEDI_MAX_VALUE,
        ]

    return True
