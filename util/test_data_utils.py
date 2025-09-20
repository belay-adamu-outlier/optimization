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
    initialize_csv,
    generate_dummy_envelope_results,
    generate_dummy_shading_results,
    print_optimization_table,
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

    def test_print_optimization_table_deduplicates_pareto_and_recent(self, capsys):
        """Pareto solutions should not appear twice in the table."""
        solutions_by_key = {
            'a': {
                'generation': 1,
                'iteration': 1,
                'thermal_energy_demand_intensity': 100.0,
                'average_overheating_hours': 20.0,
                'max_living_room_overheating': 30.0,
                'max_bedroom_overheating': 25.0,
                'pareto_optimal': True,
                'external_wall': 'R17 - Wall',
            },
            'b': {
                'generation': 1,
                'iteration': 2,
                'thermal_energy_demand_intensity': 110.0,
                'average_overheating_hours': 22.0,
                'max_living_room_overheating': 31.0,
                'max_bedroom_overheating': 26.0,
                'pareto_optimal': False,
                'external_wall': 'R17 - Wall',
            },
        }

        variable_names = ['external_wall']
        # Envelope, unconstrained => multiobjective True
        print_optimization_table(
            solutions_by_key,
            variable_names,
            generation=1,
            constrained_analysis=False,
            overheating_criterion='ASHRAE',
            problem_type='envelope',
        )

        out = capsys.readouterr().out
        # Row for Gen=1 Iter=1 has fixed-width tokens: '1  ' and '1   ' joined with a space
        assert out.count('1   1   ') == 1

    def test_print_optimization_table_shading_formatting(self, capsys):
        """Shading table should include Total_Shading_Area and Pareto mark once."""
        solutions_by_key = {
            's1': {
                'generation': 3,
                'iteration': 5,
                'thermal_energy_demand_intensity': 95.0,
                'max_living_room_overheating': 15.0,
                'max_bedroom_overheating': 12.0,
                'total_shading_area': 123.456,
                'pareto_optimal': True,
                'shading_schedule': [1, 0] * 12,
            }
        }

        print_optimization_table(
            solutions_by_key,
            variable_names=['Shading_Schedule'],
            generation=3,
            constrained_analysis=False,
            overheating_criterion='ASHRAE',
            problem_type='shading',
        )

        out = capsys.readouterr().out
        assert 'Total_Shading_Area' in out
        # Value formatted to 2 decimals should appear
        assert '123.46' in out


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
            'key1': {'pareto_optimal': False, 'feasible': True},
            'key2': {'pareto_optimal': False, 'feasible': True},
            'key3': {'pareto_optimal': False, 'feasible': True}
        }
        
        all_objectives = [
            [100.0, 25.0],  # Solution 1: dominated by solution 2
            [80.0, 20.0],   # Solution 2: dominates solution 1
            [90.0, 15.0]    # Solution 3: non-dominated
        ]
        
        update_pareto_optimality_for_display(solutions_by_key, all_objectives, constrained_analysis=False)
        
        # Solution 1 should not be Pareto optimal (dominated by solution 2)
        assert not solutions_by_key['key1']['pareto_optimal']
        # Solution 2 should be Pareto optimal
        assert solutions_by_key['key2']['pareto_optimal']
        # Solution 3 should be Pareto optimal
        assert solutions_by_key['key3']['pareto_optimal']

    def test_update_pareto_optimality_for_display_constrained_filters_infeasible(self):
        """Pareto should be computed only among feasible solutions when constrained."""
        solutions_by_key = {
            'key1': {'pareto_optimal': False, 'feasible': False},  # infeasible but would dominate
            'key2': {'pareto_optimal': False, 'feasible': True},
            'key3': {'pareto_optimal': False, 'feasible': True}
        }

        # key1 is best on both objectives but infeasible; should not be Pareto
        all_objectives = [
            [50.0, 10.0],   # key1 (infeasible)
            [80.0, 20.0],   # key2
            [90.0, 15.0]    # key3
        ]

        update_pareto_optimality_for_display(solutions_by_key, all_objectives, constrained_analysis=True)

        assert not solutions_by_key['key1']['pareto_optimal']
        assert solutions_by_key['key2']['pareto_optimal']
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


class TestDummyDataGeneration:
    """Test cases for dummy data generation functions."""
    
    def test_generate_dummy_envelope_results_ashrae(self):
        """Test ASHRAE dummy envelope results generation."""
        params = {
            'external_wall': 'R17 - Wall',
            'roof': 'R70 - Roof',
            'window': 'Trp LoE Film'
        }
        
        results = generate_dummy_envelope_results(params, 'ASHRAE')
        
        assert results['success'] is True
        assert 'district_heating_kwh' in results
        assert 'average_overheating_hours' in results
        assert 'max_overheating_hours' in results
        assert results['max_living_room_overheating'] is None
        assert results['max_bedroom_overheating'] is None
        
        # Check that values are reasonable
        # With test params: 'R17 - Wall'(12) + 'R70 - Roof'(9) + 'Trp LoE Film'(13) = 34 chars
        # So district_heating = random(500,2000) + 34*100 = random(500,2000) + 3400 = 3900-5400
        # average_overheating = random(0,100) + 34*5 = random(0,100) + 170 = 170-270
        assert 3900 <= results['district_heating_kwh'] <= 5400
        assert 170 <= results['average_overheating_hours'] <= 270
        assert 0 <= results['max_overheating_hours'] <= 200
    
    def test_generate_dummy_envelope_results_cibse59(self):
        """Test CIBSE59 dummy envelope results generation."""
        params = {
            'external_wall': 'R17 - Wall',
            'roof': 'R70 - Roof',
            'window': 'Trp LoE Film'
        }
        
        results = generate_dummy_envelope_results(params, 'CIBSE59')
        
        assert results['success'] is True
        assert 'district_heating_kwh' in results
        assert 'average_overheating_hours' in results
        assert 'max_overheating_hours' in results
        assert 'max_living_room_overheating' in results
        assert 'max_bedroom_overheating' in results
        
        # Check that values are reasonable
        # With test params: 'R17 - Wall'(12) + 'R70 - Roof'(9) + 'Trp LoE Film'(13) = 34 chars
        # So district_heating = random(500,2000) + 34*100 = random(500,2000) + 3400 = 3900-5400
        # average_overheating = random(0,50) + 34*2 = random(0,50) + 68 = 68-118
        assert 3900 <= results['district_heating_kwh'] <= 5400
        assert 68 <= results['average_overheating_hours'] <= 118
        assert 15 <= results['max_living_room_overheating'] <= 80
        assert 15 <= results['max_bedroom_overheating'] <= 80
    
    def test_generate_dummy_shading_results(self):
        """Test dummy shading results generation."""
        shading_schedule = [1, 0, 1, 0] * 6  # 24 values
        
        results = generate_dummy_shading_results(shading_schedule)
        
        assert results['success'] is True
        assert 'district_heating_kwh' in results
        assert 'average_overheating_hours' in results
        assert 'max_living_room_overheating' in results
        assert 'max_bedroom_overheating' in results
        
        # Check that values are reasonable
        assert 70 <= results['district_heating_kwh'] / 2590.29 <= 120  # TEDI range
        assert 0 <= results['average_overheating_hours'] <= 50
        assert 0 <= results['max_living_room_overheating'] <= 50
        assert 0 <= results['max_bedroom_overheating'] <= 50


if __name__ == '__main__':
    pytest.main([__file__])
