#!/usr/bin/env python3
"""
Unit tests for simulation_utils.py

This module contains comprehensive tests for all functions in simulation_utils.py:
- Temperature data extraction
- District heating extraction
- Shared cache management
"""

import unittest
import tempfile
import os
import json
import pandas as pd
from datetime import datetime
from unittest.mock import patch, mock_open
# Import the functions to test
from .simulation_utils import (
    extract_zone_temperatures,
    extract_district_heating_from_htm,
    create_shared_cache,
    make_cache_key,
    set_in_cache,
    get_from_cache,
    get_cache_entry,
    get_generation_from_cache,
    save_cache_to_file,
    compute_shading_penalty,
)
from constants.shading_variables import SHADING_VARIABLE_BOUNDS


class TestTemperatureExtraction(unittest.TestCase):
    """Test temperature data extraction functions."""
    
    def test_extract_zone_temperatures_basic(self):
        """Test basic zone temperature extraction."""
        temperature_data = {
            'zone_SW:BR': [22.1, 21.8, 21.5],
            'outdoor': [15.5, 14.2, 13.8]
        }
        timestamps = [
            datetime(2021, 1, 1, 0, 0, 0),
            datetime(2021, 1, 1, 1, 0, 0),
            datetime(2021, 1, 1, 2, 0, 0)
        ]
        
        result = extract_zone_temperatures(temperature_data, timestamps)
        
        self.assertIn('SW:BR', result)
        self.assertIn('outdoor', result)
        
        # Check zone DataFrame
        zone_df = result['SW:BR']
        self.assertIsInstance(zone_df, pd.DataFrame)
        self.assertEqual(len(zone_df), 3)
        self.assertListEqual(list(zone_df.columns), ['timestamp', 'temperature', 'date', 'month', 'day', 'hour'])
        
        # Check outdoor DataFrame
        outdoor_df = result['outdoor']
        self.assertIsInstance(outdoor_df, pd.DataFrame)
        self.assertEqual(len(outdoor_df), 3)
    
    def test_extract_zone_temperatures_no_outdoor(self):
        """Test zone temperature extraction without outdoor data."""
        temperature_data = {
            'zone_SW:BR': [22.1, 21.8, 21.5]
        }
        timestamps = [
            datetime(2021, 1, 1, 0, 0, 0),
            datetime(2021, 1, 1, 1, 0, 0),
            datetime(2021, 1, 1, 2, 0, 0)
        ]
        
        result = extract_zone_temperatures(temperature_data, timestamps)
        
        self.assertIn('SW:BR', result)
        self.assertNotIn('outdoor', result)
    
    def test_extract_zone_temperatures_empty(self):
        """Test zone temperature extraction with empty data."""
        temperature_data = {}
        timestamps = []
        
        result = extract_zone_temperatures(temperature_data, timestamps)
        
        self.assertEqual(result, {})


class TestDistrictHeatingExtraction(unittest.TestCase):
    """Test district heating extraction from HTML files."""
    
    def test_extract_district_heating_basic(self):
        """Test basic district heating extraction."""
        html_content = """
        <table>
            <tr>
                <td>Total End Uses</td>
                <td>100.0</td>
                <td>200.0</td>
                <td>300.0</td>
                <td>400.0</td>
                <td>500.0</td>
                <td>600.0</td>
                <td>700.0</td>
                <td>800.0</td>
                <td>900.0</td>
                <td>1000.0</td>
                <td>1100.0</td>
                <td>1234.56</td>
            </tr>
        </table>
        """
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.htm') as f:
            f.write(html_content)
            htm_file = f.name
        
        try:
            result = extract_district_heating_from_htm(htm_file)
            self.assertEqual(result, 1234.56)
            
        finally:
            os.unlink(htm_file)
    
    def test_extract_district_heating_file_not_found(self):
        """Test district heating extraction with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            extract_district_heating_from_htm("nonexistent_file.htm")
    
    def test_extract_district_heating_no_match(self):
        """Test district heating extraction with no matching data."""
        html_content = """
        <table>
            <tr>
                <td>Other Data</td>
                <td>100.0</td>
            </tr>
        </table>
        """
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.htm') as f:
            f.write(html_content)
            htm_file = f.name
        
        try:
            with self.assertRaises(ValueError):
                extract_district_heating_from_htm(htm_file)
                
        finally:
            os.unlink(htm_file)
    
    def test_extract_district_heating_invalid_value(self):
        """Test district heating extraction with invalid numeric value."""
        html_content = """
        <table>
            <tr>
                <td>Total End Uses</td>
                <td>100.0</td>
                <td>200.0</td>
                <td>300.0</td>
                <td>400.0</td>
                <td>500.0</td>
                <td>600.0</td>
                <td>700.0</td>
                <td>800.0</td>
                <td>900.0</td>
                <td>1000.0</td>
                <td>1100.0</td>
                <td>invalid</td>
            </tr>
        </table>
        """
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.htm') as f:
            f.write(html_content)
            htm_file = f.name
        
        try:
            with self.assertRaises(ValueError):
                extract_district_heating_from_htm(htm_file)
                
        finally:
            os.unlink(htm_file)


class TestSharedCacheFunctions(unittest.TestCase):
    """Test shared cache management functions."""
    
    def test_create_shared_cache(self):
        """Test shared cache creation."""
        cache = create_shared_cache()
        self.assertIsNotNone(cache)
        self.assertEqual(len(cache), 0)
    
    def test_make_cache_key(self):
        """Test cache key creation."""
        params = {
            'external_wall': 'R3 - Wall',
            'roof': 'R3 - Roof',
            'wwr_east': 0.2
        }
        
        key = make_cache_key(params)
        expected_key = "('R3 - Wall', 'R3 - Roof', 0.2)"
        self.assertEqual(key, expected_key)
    
    def test_set_in_cache_basic(self):
        """Test basic cache setting."""
        cache = create_shared_cache()
        params = {'test': 'value'}
        key = make_cache_key(params)
        value = [100.0, 50.0]
        
        set_in_cache(cache, key, value, max_overheating_hours=0.0)
        
        self.assertIn(key, cache)
        self.assertEqual(cache[key], value)
    
    def test_set_in_cache_with_generation(self):
        """Test cache setting with generation info and parameters."""
        cache = create_shared_cache()
        params = {'test': 'value', 'param2': 123}
        key = make_cache_key(params)
        value = [100.0, 50.0]
        generation = 5
        
        set_in_cache(cache, key, value, max_overheating_hours=0.0, generation=generation, params=params)
        
        self.assertIn(key, cache)
        cached_entry = cache[key]
        self.assertIsInstance(cached_entry, dict)
        self.assertEqual(cached_entry['objectives'], value)
        self.assertEqual(cached_entry['generation'], generation)
        # Params should be stored when provided
        self.assertEqual(cached_entry['params'], params)
    
    def test_get_from_cache_basic(self):
        """Test basic cache retrieval."""
        cache = create_shared_cache()
        params = {'test': 'value'}
        key = make_cache_key(params)
        value = [100.0, 50.0]
        
        set_in_cache(cache, key, value, max_overheating_hours=0.0)
        retrieved, max_overheating = get_from_cache(cache, key)
        
        self.assertEqual(retrieved, value)
        self.assertEqual(max_overheating, -1)  # Simple storage returns -1 for max_overheating
    
    def test_get_from_cache_with_generation(self):
        """Test cache retrieval with generation info."""
        cache = create_shared_cache()
        params = {'test': 'value'}
        key = make_cache_key(params)
        value = [100.0, 50.0]
        generation = 5
        
        set_in_cache(cache, key, value, max_overheating_hours=0.0, generation=generation, params=params)
        retrieved, max_overheating = get_from_cache(cache, key)
        
        self.assertEqual(retrieved, value)
        self.assertEqual(max_overheating, 0.0)
    
    def test_get_from_cache_not_found(self):
        """Test cache retrieval for non-existent key."""
        cache = create_shared_cache()
        result = get_from_cache(cache, 'nonexistent')
        
        self.assertIsNone(result)
    
    def test_get_cache_entry_basic(self):
        """Test getting full cache entry."""
        cache = create_shared_cache()
        params = {'test': 'value'}
        key = make_cache_key(params)
        value = [100.0, 50.0]
        generation = 5
        
        set_in_cache(cache, key, value, max_overheating_hours=0.0, generation=generation, params=params)
        entry = get_cache_entry(cache, key)
        
        self.assertIsInstance(entry, dict)
        self.assertEqual(entry['objectives'], value)
        self.assertEqual(entry['generation'], generation)
        # Params should be stored when provided
        self.assertEqual(entry['params'], params)
    
    def test_get_generation_from_cache(self):
        """Test getting generation from cache."""
        cache = create_shared_cache()
        params = {'test': 'value'}
        key = make_cache_key(params)
        value = [100.0, 50.0]
        generation = 5
        
        set_in_cache(cache, key, value, max_overheating_hours=0.0, generation=generation, params=params)
        retrieved_generation = get_generation_from_cache(cache, key)
        
        self.assertEqual(retrieved_generation, generation)
    
    def test_get_generation_from_cache_old_format(self):
        """Test getting generation from cache with old format (no generation info)."""
        cache = create_shared_cache()
        params = {'test': 'value'}
        key = make_cache_key(params)
        value = [100.0, 50.0]
        
        set_in_cache(cache, key, value, max_overheating_hours=0.0)  # No generation
        retrieved_generation = get_generation_from_cache(cache, key)
        
        self.assertIsNone(retrieved_generation)
    
    def test_get_generation_from_cache_not_found(self):
        """Test getting generation for non-existent key."""
        cache = create_shared_cache()
        result = get_generation_from_cache(cache, 'nonexistent')
        
        self.assertIsNone(result)
    
    def test_save_cache_to_file_basic(self):
        """Test basic cache saving to file."""
        cache = create_shared_cache()
        params1 = {'test1': 'value1'}
        params2 = {'test2': 'value2'}
        key1 = make_cache_key(params1)
        key2 = make_cache_key(params2)
        
        set_in_cache(cache, key1, [100.0, 50.0], 1, generation=1, params=params1)
        set_in_cache(cache, key2, [200.0, 75.0], 2, generation=1, params=params2)
        
        # Mock file operations to avoid actual file creation
        with patch('os.makedirs') as mock_makedirs, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_json_dump, \
             patch('os.path.exists', return_value=False):
            
            # Call the function
            save_cache_to_file(cache, 5, "test_debug_dir")
            
            # Verify directory creation
            mock_makedirs.assert_called_once_with("test_debug_dir", exist_ok=True)
            
            # Verify file opening (should be called twice: once for reading, once for writing)
            expected_filename = "cache_debug.json"
            expected_filepath = os.path.join("test_debug_dir", expected_filename)
            self.assertEqual(mock_file.call_count, 1)  # Only write call since file doesn't exist
            mock_file.assert_called_with(expected_filepath, 'w')
            
            # Verify JSON dump was called with the cache data
            mock_json_dump.assert_called_once()
            args, kwargs = mock_json_dump.call_args
            cache_dict = args[0]  # First argument should be the cache dict
            self.assertEqual(len(cache_dict), 2)
            self.assertIn(key1, cache_dict)
            self.assertIn(key2, cache_dict)
            self.assertEqual(cache_dict[key1]['objectives'], [100.0, 50.0])
            self.assertEqual(cache_dict[key2]['objectives'], [200.0, 75.0])
    
    def test_save_cache_to_file_empty_cache(self):
        """Test saving empty cache to file."""
        cache = create_shared_cache()
        
        with patch('os.makedirs') as mock_makedirs, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_json_dump, \
             patch('os.path.exists', return_value=False):
            
            save_cache_to_file(cache, 1)
            
            # Verify JSON dump was called with empty dict
            mock_json_dump.assert_called_once()
            args, kwargs = mock_json_dump.call_args
            cache_dict = args[0]
            self.assertEqual(len(cache_dict), 0)
    
    def test_save_cache_to_file_update_existing(self):
        """Test updating an existing cache file with new entries."""
        cache = create_shared_cache()
        params1 = {'test1': 'value1'}
        params2 = {'test2': 'value2'}
        key1 = make_cache_key(params1)
        key2 = make_cache_key(params2)
        
        set_in_cache(cache, key1, [100.0, 50.0], 1, generation=1, params=params1)
        set_in_cache(cache, key2, [200.0, 75.0], 2, generation=1, params=params2)
        
        # Mock existing cache file content
        existing_cache_content = json.dumps({
            key1: {'objectives': [100.0, 50.0], 'generation': 1},
            'existing_key': {'objectives': [300.0, 150.0], 'generation': 0}
        })
        
        with patch('os.makedirs') as mock_makedirs, \
             patch('builtins.open', mock_open(read_data=existing_cache_content)) as mock_file, \
             patch('json.dump') as mock_json_dump, \
             patch('os.path.exists', return_value=True):
            
            # Call the function
            save_cache_to_file(cache, 2, "test_debug_dir")
            
            # Verify file was opened for reading and writing
            expected_filename = "cache_debug.json"
            expected_filepath = os.path.join("test_debug_dir", expected_filename)
            
            # Should be called twice: once for reading, once for writing
            self.assertEqual(mock_file.call_count, 2)
            mock_file.assert_any_call(expected_filepath, 'r')
            mock_file.assert_any_call(expected_filepath, 'w')
            
            # Verify JSON dump was called with merged cache data
            mock_json_dump.assert_called_once()
            args, kwargs = mock_json_dump.call_args
            cache_dict = args[0]
            self.assertEqual(len(cache_dict), 3)  # 2 existing + 1 new
            self.assertIn(key1, cache_dict)
            self.assertIn(key2, cache_dict)
            self.assertIn('existing_key', cache_dict)
            self.assertEqual(cache_dict[key2]['objectives'], [200.0, 75.0])  # New entry
    
    def test_save_cache_to_file_exception_handling(self):
        """Test that exceptions during file saving are handled gracefully."""
        cache = create_shared_cache()
        params = {'test': 'value'}
        key = make_cache_key(params)
        set_in_cache(cache, key, [100.0, 50.0], 1)
        
        # Mock os.makedirs to raise an exception
        with patch('os.makedirs', side_effect=PermissionError("Permission denied")), \
             patch('builtins.print') as mock_print:
            
            # Should not raise an exception
            save_cache_to_file(cache, 1)
            
            # Should print a warning message
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            self.assertIn("Warning: Could not save debug cache file", call_args)


class TestPenaltyFunction(unittest.TestCase):
    """Tests for the shading penalty objective function."""

    def test_penalty_zero_when_all_zero(self):
        params = {k: 0.0 for k in SHADING_VARIABLE_BOUNDS.keys()}
        penalty = compute_shading_penalty(
            params, SHADING_VARIABLE_BOUNDS, 'shading'
        )
        self.assertAlmostEqual(penalty, 0.0, places=8)

    def test_penalty_quadratic_normalized(self):
        # Use half of max for three variables -> each contributes (0.5)^2/(1)^2
        # since normalization divides by max^2, the term is (v/max)^2.
        params = {
            'south_overhang_depth': SHADING_VARIABLE_BOUNDS[
                'south_overhang_depth'
            ]['max'] * 0.5,
            'south_overhang_angle': SHADING_VARIABLE_BOUNDS[
                'south_overhang_angle'
            ]['max'] * 0.5,
            'west_right_fin_depth': SHADING_VARIABLE_BOUNDS[
                'west_right_fin_depth'
            ]['max'] * 0.5,
        }
        expected = 3 * (0.5 ** 2)
        penalty = compute_shading_penalty(
            params, SHADING_VARIABLE_BOUNDS, 'shading'
        )
        self.assertAlmostEqual(penalty, expected, places=8)

    def test_penalty_ignores_negative_values(self):
        params = {'south_left_fin_depth': -1.0}
        penalty = compute_shading_penalty(
            params, SHADING_VARIABLE_BOUNDS, 'shading'
        )
        self.assertAlmostEqual(penalty, 0.0, places=8)


if __name__ == '__main__':
    unittest.main() 