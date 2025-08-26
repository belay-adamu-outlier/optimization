#!/usr/bin/env python3
"""
Test scenarios for cli_utils.py functions using pytest.
Tests the CLI interaction functions including the new test_mode prompt.
"""

import pytest
from unittest.mock import patch, Mock
from .cli_utils import (
    get_interactive_arguments,
    _get_boolean_input,
    _get_weather_file_choice,
    _get_numeric_input,
    _print_summary_and_confirm
)


class TestCLIUtilities:
    """Test cases for CLI utility functions."""
    
    def test_get_boolean_input_true(self):
        """Test boolean input returning True."""
        with patch('builtins.input', return_value='y'):
            result = _get_boolean_input("Test prompt", False, "test")
            assert result is True
    
    def test_get_boolean_input_false(self):
        """Test boolean input returning False."""
        with patch('builtins.input', return_value='n'):
            result = _get_boolean_input("Test prompt", False, "test")
            assert result is False
    
    def test_get_boolean_input_default(self):
        """Test boolean input using default value."""
        with patch('builtins.input', return_value=''):
            result = _get_boolean_input("Test prompt", True, "test")
            assert result is True
    
    def test_get_weather_file_choice(self):
        """Test weather file choice selection."""
        with patch('builtins.input', return_value='1'):
            result = _get_weather_file_choice()
            assert result == "Vancouver - 2021 AMY.epw"
    
    def test_get_numeric_input(self):
        """Test numeric input handling."""
        with patch('builtins.input', return_value='50'):
            result = _get_numeric_input("Test prompt", 100, "test")
            assert result == 50
    
    def test_get_numeric_input_default(self):
        """Test numeric input using default value."""
        with patch('builtins.input', return_value=''):
            result = _get_numeric_input("Test prompt", 100, "test")
            assert result == 100
    
    @patch('builtins.input')
    def test_get_interactive_arguments_complete(self, mock_input):
        """Test complete interactive argument collection."""
        # Mock all user inputs
        mock_input.side_effect = [
            'y',  # test_mode
            '1',  # weather_file
            'y',  # constrained_analysis
            '2',  # overheating_criterion (CIBSE59)
            '50', # n_generations
            '100', # population_size
            '0.7', # crossover_prob
            '0.1', # mutation_prob
            'y',   # debug_mode
            '12345', # seed
            'y'   # confirm
        ]
        
        with patch('builtins.print'):  # Suppress print output during testing
            args = get_interactive_arguments()
        
        assert args['test_mode'] is True
        assert args['weather'] == "Vancouver - 2021 AMY.epw"
        assert args['constrained_analysis'] is True
        assert args['overheating_criterion'] == 'CIBSE59'
        assert args['generations'] == 50
        assert args['population_size'] == 100
        assert args['crossover_prob'] == 0.7
        assert args['mutation_prob'] == 0.1
        assert args['seed'] == 12345
    
    @patch('builtins.input')
    def test_get_interactive_arguments_defaults(self, mock_input):
        """Test interactive argument collection with default values."""
        # Mock all user inputs to use defaults
        mock_input.side_effect = [
            'n',  # test_mode
            '',   # weather_file (use default)
            'n',  # constrained_analysis
            '',   # overheating_criterion (use default)
            '',   # n_generations (use default)
            '',   # population_size (use default)
            '',   # crossover_prob (use default)
            '',   # mutation_prob (use default)
            'n',  # debug_mode
            '',   # seed (use default)
            'y'   # confirm
        ]
        
        with patch('builtins.print'):  # Suppress print output during testing
            args = get_interactive_arguments()
        
        assert args['test_mode'] is False
        assert args['weather'] == "Vancouver - 2021 AMY.epw"
        assert args['constrained_analysis'] is False
        assert args['overheating_criterion'] == 'ASHRAE'
        assert args['generations'] == 100
        assert args['population_size'] == 50
        assert args['crossover_prob'] == 0.5
        assert args['mutation_prob'] == 0.125
        assert args['seed'] is None
    
    @patch('builtins.input')
    def test_get_interactive_arguments_test_mode_only(self, mock_input):
        """Test interactive argument collection with only test_mode enabled."""
        # Mock user inputs - only enable test_mode, use defaults for rest
        mock_input.side_effect = [
            'y',  # test_mode
            '',   # weather_file (use default)
            'n',  # constrained_analysis
            '',   # overheating_criterion (use default)
            '',   # n_generations (use default)
            '',   # population_size (use default)
            '',   # crossover_prob (use default)
            '',   # mutation_prob (use default)
            'n',  # debug_mode
            '',   # seed (use default)
            'y'   # confirm
        ]
        
        with patch('builtins.print'):  # Suppress print output during testing
            args = get_interactive_arguments()
        
        assert args['test_mode'] is True
        assert args['weather'] == "Vancouver - 2021 AMY.epw"
        assert args['constrained_analysis'] is False
        assert args['overheating_criterion'] == 'ASHRAE'
    
    def test_print_summary_and_confirm(self):
        """Test summary printing and confirmation."""
        with patch('builtins.input', return_value='y'), \
             patch('builtins.print') as mock_print:
            _print_summary_and_confirm(
                weather_file='test.epw',
                constrained_analysis=True,
                overheating_criterion='CIBSE59',
                generations=50,
                population_size=100,
                crossover_prob=0.5,
                mutation_prob=0.125,
                debug_mode=False,
                seed=None,
                test_mode=True
            )
        
        # Verify that test_mode was included in summary
        mock_print.assert_any_call("Test Mode: Enabled")
        mock_print.assert_any_call("Weather File: test.epw")
        mock_print.assert_any_call("Constrained Analysis: Enabled")
        mock_print.assert_any_call("Overheating Criterion: CIBSE59")


if __name__ == '__main__':
    pytest.main([__file__])
