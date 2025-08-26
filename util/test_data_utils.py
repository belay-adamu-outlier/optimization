#!/usr/bin/env python3
"""
Test scenarios for data_utils.py functions using pytest.
Tests the new envelope optimization plotting functions and other utility functions.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock
import tempfile
import os
import json
from .data_utils import (
    initialize_envelope_optimization_plot,
    update_envelope_optimization_plot,
    load_optimization_variables,
    load_config,
    print_generation_summary,
    print_optimization_parameters,
    print_final_results,
    x_to_params,
    update_pareto_optimality_for_display,
    initialize_csv
)


class TestEnvelopeOptimizationPlotting:
    """Test cases for envelope optimization plotting functions."""
    
    def setup_method(self):
        """Set up test data for each test."""
        self.sample_solutions = {
            'key1': {
                'generation': 1,
                'iteration': 1,
                'thermal_energy_demand_intensity': 100.0,
                'average_overheating_hours': 25.0,
                'pareto_optimal': True,
                'external_wall': 'R17 - Wall',
                'roof': 'R18 - Roof',
                'window': 'Dbl LoE Spec'
            },
            'key2': {
                'generation': 1,
                'iteration': 2,
                'thermal_energy_demand_intensity': 150.0,
                'average_overheating_hours': 35.0,
                'pareto_optimal': False,
                'external_wall': 'R17 - Wall',
                'roof': 'R18 - Roof',
                'window': 'Trp LoE'
            }
        }
    
    def test_initialize_envelope_optimization_plot_multiobjective(self):
        """Test initialization of multi-objective envelope optimization plot."""
        fig, ax, scatter_all, scatter_pareto = initialize_envelope_optimization_plot(
            constrained_analysis=False,
            overheating_criterion='ASHRAE'
        )
        
        assert fig is not None
        assert ax is not None
        assert scatter_all is not None
        assert scatter_pareto is not None
        
        # Check plot title and labels
        assert 'Multi-Objective' in ax.get_title()
        assert 'Thermal Energy Demand Intensity' in ax.get_xlabel()
        assert 'Average Overheating Hours' in ax.get_ylabel()
        
        plt.close(fig)
    
    def test_initialize_envelope_optimization_plot_constrained_ashrae(self):
        """Test initialization of ASHRAE constrained envelope optimization plot."""
        fig, ax, scatter_all, scatter_pareto = initialize_envelope_optimization_plot(
            constrained_analysis=True,
            overheating_criterion='ASHRAE'
        )
        
        assert fig is not None
        assert ax is not None
        
        # Check plot title and labels
        assert 'ASHRAE Constrained' in ax.get_title()
        assert 'Max Overheating Hours' in ax.get_ylabel()
        
        plt.close(fig)
    
    def test_initialize_envelope_optimization_plot_constrained_cibse59(self):
        """Test initialization of CIBSE59 constrained envelope optimization plot."""
        fig, ax, scatter_all, scatter_pareto = initialize_envelope_optimization_plot(
            constrained_analysis=True,
            overheating_criterion='CIBSE59'
        )
        
        assert fig is not None
        assert ax is not None
        
        # Check plot title and labels
        assert 'CIBSE59 Constrained' in ax.get_title()
        assert 'Max Overheating Hours' in ax.get_ylabel()
        
        plt.close(fig)
    
    def test_update_envelope_optimization_plot_with_solutions(self):
        """Test updating the plot with new solutions."""
        fig, ax, scatter_all, scatter_pareto = initialize_envelope_optimization_plot(
            constrained_analysis=False,
            overheating_criterion='ASHRAE'
        )
        
        # Update plot with solutions
        update_envelope_optimization_plot(
            self.sample_solutions,
            generation=1,
            fig=fig,
            ax=ax,
            scatter_all=scatter_all,
            scatter_pareto=scatter_pareto,
            constrained_analysis=False,
            overheating_criterion='ASHRAE'
        )
        
        # Check that title was updated with generation info
        assert 'Generation 1' in ax.get_title()
        
        plt.close(fig)
    
    def test_update_envelope_optimization_plot_empty_solutions(self):
        """Test updating plot with empty solutions (should not crash)."""
        fig, ax, scatter_all, scatter_pareto = initialize_envelope_optimization_plot(
            constrained_analysis=False,
            overheating_criterion='ASHRAE'
        )
        
        # Should not raise any exceptions
        update_envelope_optimization_plot(
            {},
            generation=1,
            fig=fig,
            ax=ax,
            scatter_all=scatter_all,
            scatter_pareto=scatter_pareto,
            constrained_analysis=False,
            overheating_criterion='ASHRAE'
        )
        
        plt.close(fig)


class TestLoadFunctions:
    """Test cases for loading functions."""
    
    def setup_method(self):
        """Set up temporary files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """Clean up temporary files."""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_optimization_variables_success(self):
        """Test successful loading of optimization variables."""
        # Create test file
        test_data = {
            "external_wall": ["R3 - Wall", "R9 - Wall"],
            "roof": ["R3 - Roof", "R10 - Roof"],
            "window": ["Dbl LoE", "Trp LoE"]
        }
        
        with open('optimization_variables.json', 'w') as f:
            json.dump(test_data, f)
        
        result = load_optimization_variables()
        assert result == test_data
    
    def test_load_optimization_variables_file_not_found(self):
        """Test that FileNotFoundError is raised when file doesn't exist."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_optimization_variables()
        
        assert "Optimization variables file 'optimization_variables.json' not found" in str(exc_info.value)
    
    def test_load_config_success(self):
        """Test successful loading of config."""
        # Create test file
        test_config = {"num_workers": 4}
        
        with open('config.json', 'w') as f:
            json.dump(test_config, f)
        
        result = load_config()
        assert result == test_config
    
    def test_load_config_file_not_found_creates_default(self):
        """Test that default config is created when file doesn't exist."""
        result = load_config()
        
        # Should create default config
        assert result == {"num_workers": 1}
        
        # File should now exist
        assert os.path.exists('config.json')
        
        # Verify file contents
        with open('config.json', 'r') as f:
            file_content = json.load(f)
        assert file_content == {"num_workers": 1}


class TestPrintFunctions:
    """Test cases for print functions."""
    
    def test_print_generation_summary(self, capsys):
        """Test generation summary printing."""
        print_generation_summary(
            generation=5,
            total_evaluations=20,
            new_evaluations=15,
            skipped_evaluations=5,
            cache_size=100
        )
        
        captured = capsys.readouterr()
        assert "Generation 5 Summary:" in captured.out
        assert "Total individuals in generation: 20" in captured.out
        assert "New unique evaluations: 15" in captured.out
        assert "Duplicate evaluations skipped: 5" in captured.out
        assert "Total unique solutions in cache: 100" in captured.out
    
    def test_print_optimization_parameters(self, capsys):
        """Test optimization parameters printing."""
        params = {
            "population_size": 50,
            "generations": 100,
            "crossover_prob": 0.8
        }
        
        print_optimization_parameters(params, "NSGA-II")
        
        captured = capsys.readouterr()
        assert "Optimization Parameters:" in captured.out
        assert "population_size: 50" in captured.out
        assert "generations: 100" in captured.out
        assert "crossover_prob: 0.8" in captured.out
        assert "Algorithm: NSGA-II" in captured.out
    
    def test_print_final_results(self, capsys):
        """Test final results printing."""
        # Mock result object
        mock_result = Mock()
        mock_result.X = [1, 2, 3]
        mock_result.F = [100.0, 25.0]
        mock_result.algorithm.evaluator.n_eval = 500
        
        # Mock shared cache
        mock_cache = {'key1': 'value1', 'key2': 'value2'}
        
        print_final_results(mock_result, mock_cache)
        
        captured = capsys.readouterr()
        assert "FINAL RESULTS" in captured.out
        assert "Best solution found: [1, 2, 3]" in captured.out
        assert "Function values: [100.0, 25.0]" in captured.out
        assert "Number of evaluations: 500" in captured.out
        assert "Total cached combinations: 2" in captured.out


class TestParameterConversion:
    """Test cases for parameter conversion functions."""
    
    def test_x_to_params_basic(self):
        """Test basic parameter conversion."""
        optimization_variables = {
            "external_wall": ["R3 - Wall", "R9 - Wall"],
            "roof": ["R3 - Roof", "R10 - Roof"],
            "window": ["Dbl LoE", "Trp LoE"]
        }
        
        x = [0, 1, 0]  # Select first wall, second roof, first window
        
        result = x_to_params(x, optimization_variables)
        
        expected = {
            "external_wall": "R3 - Wall",
            "roof": "R10 - Roof",
            "window": "Dbl LoE"
        }
        
        assert result == expected
    
    def test_x_to_params_with_wwr(self):
        """Test parameter conversion with WWR values (should convert to decimal)."""
        optimization_variables = {
            "wwr_south": [20, 30, 40],  # Percentages
            "external_wall": ["R3 - Wall", "R9 - Wall"]
        }
        
        x = [1, 0]  # Select 30% WWR, first wall
        
        result = x_to_params(x, optimization_variables)
        
        expected = {
            "wwr_south": 0.30,  # Should be converted to decimal
            "external_wall": "R3 - Wall"
        }
        
        assert result == expected


class TestParetoOptimality:
    """Test cases for Pareto optimality functions."""
    
    def test_update_pareto_optimality_for_display(self):
        """Test Pareto optimality update for display."""
        solutions_by_key = {
            'key1': {'pareto_optimal': False},
            'key2': {'pareto_optimal': False},
            'key3': {'pareto_optimal': False}
        }
        
        all_objectives = [
            [100.0, 25.0],  # Solution 1: dominated by solution 2
            [80.0, 20.0],   # Solution 2: dominates solution 1
            [90.0, 15.0]    # Solution 3: non-dominated
        ]
        
        update_pareto_optimality_for_display(solutions_by_key, all_objectives)
        
        # Solution 1 should not be Pareto optimal (dominated by solution 2)
        assert not solutions_by_key['key1']['pareto_optimal']
        # Solution 2 should be Pareto optimal
        assert solutions_by_key['key2']['pareto_optimal']
        # Solution 3 should be Pareto optimal
        assert solutions_by_key['key3']['pareto_optimal']


class TestCSVOperations:
    """Test cases for CSV operations."""
    
    def setup_method(self):
        """Set up temporary files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """Clean up temporary files."""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialize_csv_envelope(self):
        """Test CSV initialization for envelope optimization."""
        metadata = {'Algorithm': 'NSGA-II', 'Generations': 100}
        variable_names = ['external_wall', 'roof', 'window']
        
        csv_filename = 'test_envelope.csv'
        initialize_csv(
            csv_filename,
            metadata,
            variable_names,
            problem_type='envelope',
            constrained_analysis=False,
            overheating_criterion='ASHRAE'
        )
        
        assert os.path.exists(csv_filename)
        
        # Check file contents
        with open(csv_filename, 'r') as f:
            content = f.read()
            assert 'Optimization Metadata' in content
            assert 'Algorithm: NSGA-II' in content
            assert 'Generations: 100' in content
            assert 'Generation,Iteration,external_wall,roof,window,Thermal_Energy_Demand_Intensity,Average_Overheating_Hours,Pareto_Optimal' in content
    
    def test_initialize_csv_shading(self):
        """Test CSV initialization for shading optimization."""
        metadata = {'Algorithm': 'NSGA-II', 'Generations': 100}
        objective_names = ['Thermal_Energy_Demand_Intensity', 'Max_Living_Room_Overheating']
        
        csv_filename = 'test_shading.csv'
        initialize_csv(
            csv_filename,
            metadata,
            objective_names=objective_names,
            problem_type='shading',
            constrained_analysis=False,
            overheating_criterion='CIBSE59',
            multiobjective=True
        )
        
        assert os.path.exists(csv_filename)
        
        # Check file contents
        with open(csv_filename, 'r') as f:
            content = f.read()
            assert 'Generation,Iteration,Shading_Schedule,Thermal_Energy_Demand_Intensity,Max_Living_Room_Overheating,Max_Bedroom_Overheating,Pareto_Optimal' in content


if __name__ == '__main__':
    pytest.main([__file__])
