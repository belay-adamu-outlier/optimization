#!/usr/bin/env python3
"""
Test scenarios for main.py functions using pytest.
Tests the main optimization classes and functions.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

import pytest
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch

# Import the functions we want to test
from main import (
    BuildingEnvelopeProblem,
    EvaluationCallback,
    main
)


# Mock CSV creation during testing to avoid file creation entirely
@pytest.fixture(scope="session", autouse=True)
def mock_csv_creation():
    """Mock CSV creation functions to prevent real file creation during testing."""
    with patch('main.initialize_csv'), \
         patch('main.save_to_csv'), \
         patch('main.save_final_results'):
        yield


class TestBuildingEnvelopeProblem:
    """Test cases for BuildingEnvelopeProblem class."""
    
    def setup_method(self):
        """Set up test data for each test."""
        self.optimization_variables = {
            "external_wall": ["R3 - Wall", "R9 - Wall"],
            "roof": ["R3 - Roof", "R10 - Roof"],
            "window": ["Dbl LoE", "Trp LoE"]
        }
        self.weather_file = "test_weather.epw"
        self.shared_cache = {}
    
    def test_init_multiobjective(self):
        """Test initialization of multi-objective problem."""
        problem = BuildingEnvelopeProblem(
            optimization_variables=self.optimization_variables,
            weather_file=self.weather_file,
            constrained_analysis=False,
            overheating_criterion='ASHRAE',
            test_mode=False
        )
        
        assert problem.n_var == 3
        assert problem.n_obj == 2
        assert problem.n_ieq_constr == 0
        assert problem.vtype == int
        assert problem.weather_file == self.weather_file
        assert problem.constrained_analysis is False
        assert problem.overheating_criterion == 'ASHRAE'
        assert problem.current_generation == 0
        assert problem.test_mode is False
    
    def test_init_test_mode(self):
        """Test initialization with test mode enabled."""
        problem = BuildingEnvelopeProblem(
            optimization_variables=self.optimization_variables,
            weather_file=self.weather_file,
            constrained_analysis=False,
            overheating_criterion='ASHRAE',
            test_mode=True
        )
        
        assert problem.test_mode is True
        assert problem.constrained_analysis is False
        assert problem.overheating_criterion == 'ASHRAE'
    
    def test_init_constrained_ashrae(self):
        """Test initialization of ASHRAE constrained problem."""
        problem = BuildingEnvelopeProblem(
            optimization_variables=self.optimization_variables,
            weather_file=self.weather_file,
            constrained_analysis=True,
            overheating_criterion='ASHRAE',
            test_mode=False
        )
        
        assert problem.n_var == 3
        assert problem.n_obj == 1
        assert problem.n_ieq_constr == 1
        assert problem.constrained_analysis is True
        assert problem.overheating_criterion == 'ASHRAE'
        assert problem.test_mode is False
    
    def test_init_constrained_cibse59(self):
        """Test initialization of CIBSE59 constrained problem."""
        problem = BuildingEnvelopeProblem(
            optimization_variables=self.optimization_variables,
            weather_file=self.weather_file,
            constrained_analysis=True,
            overheating_criterion='CIBSE59',
            test_mode=False
        )
        
        assert problem.n_var == 3
        assert problem.n_obj == 1
        assert problem.n_ieq_constr == 2
        assert problem.constrained_analysis is True
        assert problem.overheating_criterion == 'CIBSE59'
        assert problem.test_mode is False
    
    def test_set_generation(self):
        """Test setting generation number."""
        problem = BuildingEnvelopeProblem(
            optimization_variables=self.optimization_variables,
            weather_file=self.weather_file,
            test_mode=False
        )
        
        problem.set_generation(5)
        assert problem.current_generation == 5
    
    def test_bounds_calculation(self):
        """Test that bounds are calculated correctly."""
        problem = BuildingEnvelopeProblem(
            optimization_variables=self.optimization_variables,
            weather_file=self.weather_file,
            test_mode=False
        )
        
        # Should have 3 variables
        assert len(problem.xl) == 3
        assert len(problem.xu) == 3
        
        # All lower bounds should be 0
        assert np.all(problem.xl == 0)
        
        # Upper bounds should be (len(options) - 1) for each variable
        expected_upper_bounds = [1, 1, 1]  # 2 options each, so max index is 1
        assert np.all(problem.xu == expected_upper_bounds)
    
    @patch('main.run_simulation')
    @patch('main.analyze_simulation_results')
    @patch('main.set_in_cache')
    def test_evaluate_new_combination_multiobjective(self, mock_set_cache, mock_analyze, mock_run_sim):
        """Test evaluation of new combination in multi-objective mode."""
        # Mock simulation results
        mock_run_sim.return_value = "simulation_output_dir"
        mock_analyze.return_value = {
            'success': True,
            'district_heating_kwh': 2590.29,  # This will give TEDI = 1.0
            'average_overheating_hours': 25.0
        }
        
        problem = BuildingEnvelopeProblem(
            optimization_variables=self.optimization_variables,
            shared_cache=self.shared_cache,
            weather_file=self.weather_file,
            constrained_analysis=False,
            overheating_criterion='ASHRAE',
            test_mode=False
        )
        
        # Test vector [0, 1, 0] - first wall, second roof, first window
        x = np.array([0, 1, 0])
        out = {}
        
        problem._evaluate(x, out)
        
        # Check that objectives were set correctly
        assert len(out["F"]) == 2
        assert out["F"][0] == 1.0  # TEDI
        assert out["F"][1] == 25.0  # Overheating hours
        
        # Check that simulation was called
        mock_run_sim.assert_called_once()
        mock_analyze.assert_called_once()
        mock_set_cache.assert_called_once()
    
    @patch('main.run_simulation')
    @patch('main.analyze_simulation_results')
    @patch('main.set_in_cache')
    def test_evaluate_new_combination_constrained(self, mock_set_cache, mock_analyze, mock_run_sim):
        """Test evaluation of new combination in constrained mode."""
        # Mock simulation results
        mock_run_sim.return_value = "simulation_output_dir"
        mock_analyze.return_value = {
            'success': True,
            'district_heating_kwh': 2590.29,
            'average_overheating_hours': 25.0,  # Required for TEDI calculation
            'max_overheating_hours': 20.0  # Required for constraint calculation
        }
        
        problem = BuildingEnvelopeProblem(
            optimization_variables=self.optimization_variables,
            shared_cache=self.shared_cache,
            weather_file=self.weather_file,
            constrained_analysis=True,
            overheating_criterion='ASHRAE',
            test_mode=False
        )
        
        x = np.array([0, 1, 0])
        out = {}
        
        problem._evaluate(x, out)
        
        # Check that only one objective was set (constrained mode)
        assert len(out["F"]) == 1
        assert out["F"][0] == 1.0  # TEDI only
        
        # Check that constraints were set
        assert len(out["G"]) == 1
        assert out["G"][0] == -10.0  # 20 - 30 = -10 (within constraint)
    
    @patch('main.run_simulation')
    @patch('main.analyze_simulation_results')
    def test_evaluate_simulation_failure(self, mock_analyze, mock_run_sim):
        """Test evaluation when simulation fails."""
        mock_run_sim.return_value = "simulation_output_dir"
        mock_analyze.return_value = {
            'success': False,
            'error': 'Simulation failed'
        }
        
        problem = BuildingEnvelopeProblem(
            optimization_variables=self.optimization_variables,
            shared_cache=self.shared_cache,
            weather_file=self.weather_file,
            constrained_analysis=False,
            test_mode=False
        )
        
        x = np.array([0, 1, 0])
        out = {}
        
        problem._evaluate(x, out)
        
        # Should return penalty values
        assert len(out["F"]) == 2
        assert out["F"][0] == 1000.0
        assert out["F"][1] == 1000.0
    
    @patch('main.generate_dummy_envelope_results')
    def test_evaluate_test_mode(self, mock_generate_dummy):
        """Test evaluation when test mode is enabled."""
        # Mock dummy results
        mock_generate_dummy.return_value = {
            'success': True,
            'district_heating_kwh': 2590.29,  # This will give TEDI = 1.0
            'average_overheating_hours': 25.0
        }
        
        problem = BuildingEnvelopeProblem(
            optimization_variables=self.optimization_variables,
            shared_cache=self.shared_cache,
            weather_file=self.weather_file,
            constrained_analysis=False,
            overheating_criterion='ASHRAE',
            test_mode=True
        )
        
        x = np.array([0, 1, 0])
        out = {}
        
        problem._evaluate(x, out)
        
        # Check that dummy data was generated instead of simulation
        mock_generate_dummy.assert_called_once()
        
        # Check that objectives were set correctly
        assert len(out["F"]) == 2
        assert out["F"][0] == 1.0  # TEDI
        assert out["F"][1] == 25.0  # Overheating hours
    
    def test_evaluate_cached_result(self):
        """Test evaluation when result is already cached."""
        # Pre-populate cache
        cached_objectives = [100.0, 25.0]
        cached_max_overheating = 30.0
        
        self.shared_cache['cached_key'] = {
            'objectives': cached_objectives,
            'max_overheating_hours': cached_max_overheating
        }
        
        problem = BuildingEnvelopeProblem(
            optimization_variables=self.optimization_variables,
            shared_cache=self.shared_cache,
            weather_file=self.weather_file,
            constrained_analysis=False
        )
        
        # Mock the cache key generation to return our cached key
        with patch('main.make_cache_key', return_value='cached_key'):
            x = np.array([0, 1, 0])
            out = {}
            
            problem._evaluate(x, out)
            
            # Should return cached values without running simulation
            assert len(out["F"]) == 2
            assert out["F"][0] == 100.0
            assert out["F"][1] == 25.0
    
    def test_evaluate_exception_handling(self):
        """Test that exceptions are handled gracefully."""
        problem = BuildingEnvelopeProblem(
            optimization_variables=self.optimization_variables,
            shared_cache=self.shared_cache,
            weather_file=self.weather_file,
            constrained_analysis=False
        )
        
        # Mock x_to_params to raise an exception
        with patch('main.x_to_params', side_effect=Exception("Test error")):
            x = np.array([0, 1, 0])
            out = {}
            
            problem._evaluate(x, out)
            
            # Should return penalty values
            assert len(out["F"]) == 2
            assert out["F"][0] == 1000.0
            assert out["F"][1] == 1000.0
    
    def test_evaluate_cibse59_constraints(self):
        """Test CIBSE59 constraint handling in evaluation."""
        problem = BuildingEnvelopeProblem(
            optimization_variables=self.optimization_variables,
            shared_cache=self.shared_cache,
            weather_file=self.weather_file,
            constrained_analysis=True,
            overheating_criterion='CIBSE59'
        )
        
        with patch('main.run_simulation') as mock_run_sim, \
             patch('main.analyze_simulation_results') as mock_analyze, \
             patch('main.set_in_cache') as mock_set_cache:
            
            mock_run_sim.return_value = "simulation_output_dir"
            mock_analyze.return_value = {
                'success': True,
                'district_heating_kwh': 2590.29,
                'average_overheating_hours': 25.0,
                'max_living_room_overheating': 150.0,  # Exceeds 120 limit
                'max_bedroom_overheating': 40.0   # Exceeds 32 limit
            }
            
            x = np.array([0, 1, 0])
            out = {}
            
            problem._evaluate(x, out)
            
            # Check constraints (should be positive when exceeding limits)
            assert len(out["G"]) == 2
            assert out["G"][0] == 40.0   # 150 - 110 = 40 (constraint limit is 110, not 120)
            assert out["G"][1] == 8.0    # 40 - 32 = 8
    
    def test_evaluate_with_shared_cache_none(self):
        """Test evaluation when shared_cache is None."""
        problem = BuildingEnvelopeProblem(
            optimization_variables=self.optimization_variables,
            shared_cache=None,  # No cache
            weather_file=self.weather_file,
            constrained_analysis=False
        )
        
        with patch('main.run_simulation') as mock_run_sim, \
             patch('main.analyze_simulation_results') as mock_analyze:
            
            mock_run_sim.return_value = "simulation_output_dir"
            mock_analyze.return_value = {
                'success': True,
                'district_heating_kwh': 2590.29,
                'average_overheating_hours': 25.0
            }
            
            x = np.array([0, 1, 0])
            out = {}
            
            problem._evaluate(x, out)
            
            # Should still work without cache
            assert len(out["F"]) == 2
            assert out["F"][0] == 1.0
            assert out["F"][1] == 25.0
    
    def test_evaluate_invalid_input_types(self):
        """Test evaluation with invalid input types."""
        problem = BuildingEnvelopeProblem(
            optimization_variables=self.optimization_variables,
            shared_cache=self.shared_cache,
            weather_file=self.weather_file,
            constrained_analysis=False
        )
        
        # Test with non-numpy array
        x = [0, 1, 0]  # List instead of numpy array
        out = {}
        
        with patch('main.x_to_params') as mock_x_to_params:
            mock_x_to_params.return_value = {"external_wall": "R3 - Wall", "roof": "R10 - Roof", "window": "Dbl LoE"}
            
            with patch('main.run_simulation') as mock_run_sim, \
                 patch('main.analyze_simulation_results') as mock_analyze:
                
                mock_run_sim.return_value = "simulation_output_dir"
                mock_analyze.return_value = {
                    'success': True,
                    'district_heating_kwh': 2590.29,
                    'average_overheating_hours': 25.0
                }
                
                problem._evaluate(x, out)
                
                # Should still work
                assert len(out["F"]) == 2


class TestEvaluationCallback:
    """Test cases for EvaluationCallback class."""
    
    def setup_method(self):
        """Set up test data for each test."""
        self.optimization_variables = {
            "external_wall": ["R3 - Wall", "R9 - Wall"],
            "roof": ["R3 - Roof", "R10 - Roof"],
            "window": ["Dbl LoE", "Trp LoE"]
        }
        self.shared_cache = {}
        self.metadata = {'Algorithm': 'NSGA-II', 'Generations': 100}
    
    @patch('main.initialize_envelope_optimization_plot')
    @patch('main.initialize_csv')
    def test_init(self, mock_init_csv, mock_init_plot):
        """Test callback initialization."""
        mock_init_plot.return_value = (Mock(), Mock(), Mock(), Mock())
        
        callback = EvaluationCallback(
            optimization_variables=self.optimization_variables,
            shared_cache=self.shared_cache,
            debug_mode=True,
            metadata=self.metadata,
            constrained_analysis=False,
            overheating_criterion='ASHRAE'
        )
        
        assert callback.optimization_variables == self.optimization_variables
        assert callback.shared_cache == self.shared_cache
        assert callback.debug_mode is True
        assert callback.metadata == self.metadata
        assert callback.generation == 0
        assert callback.constrained_analysis is False
        assert callback.overheating_criterion == 'ASHRAE'
        assert callback.variable_names == list(self.optimization_variables.keys())
        
        mock_init_csv.assert_called_once()
        mock_init_plot.assert_called_once()
    
    @patch('main.print_optimization_table')
    @patch('main.save_to_csv')
    @patch('main.print_generation_summary')
    @patch('main.update_envelope_optimization_plot')
    @patch('main.save_cache_to_file')
    def test_notify_generation_complete(self, mock_save_cache, mock_update_plot, 
                                      mock_print_summary, mock_save_csv, mock_print_table):
        """Test notification when generation is complete."""
        # Mock plot initialization
        with patch('main.initialize_envelope_optimization_plot') as mock_init_plot:
            mock_init_plot.return_value = (Mock(), Mock(), Mock(), Mock())
            
            callback = EvaluationCallback(
                optimization_variables=self.optimization_variables,
                shared_cache=self.shared_cache,
                debug_mode=True,
                metadata=self.metadata
            )
            
            # Mock algorithm object
            mock_algorithm = Mock()
            mock_algorithm.pop.get.return_value = np.array([[0, 1, 0], [1, 0, 1]])
            mock_algorithm.problem.set_generation = Mock()
            
            # Call notify
            callback.notify(mock_algorithm)
            
            # Check that generation was incremented
            assert callback.generation == 1
            
            # Check that all expected functions were called
            mock_print_table.assert_called_once()
            mock_save_csv.assert_called_once()
            mock_print_summary.assert_called_once()
            mock_update_plot.assert_called_once()
            mock_save_cache.assert_called_once()
    
    @patch('main.save_final_results')
    def test_save_final_results(self, mock_save_final):
        """Test saving final results."""
        with patch('main.initialize_envelope_optimization_plot') as mock_init_plot:
            mock_init_plot.return_value = (Mock(), Mock(), Mock(), Mock())
            
            callback = EvaluationCallback(
                optimization_variables=self.optimization_variables,
                shared_cache=self.shared_cache,
                metadata=self.metadata
            )
            
            callback.save_final_results()
            
            mock_save_final.assert_called_once()


class TestMainFunction:
    """Test cases for main function."""
    
    def setup_method(self):
        """Set up test data for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        
        # Create test files in temp directory
        self.create_test_files()
    
    def teardown_method(self):
        """Clean up temporary files."""
        try:
            # Remove temporary directory
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
    
    def create_test_files(self):
        """Create test configuration files in temp directory."""
        # Create optimization_variables.json in temp directory
        optimization_vars = {
            "external_wall": ["R3 - Wall", "R9 - Wall"],
            "roof": ["R3 - Roof", "R10 - Roof"],
            "window": ["Dbl LoE", "Trp LoE"]
        }
        temp_opt_file = os.path.join(self.temp_dir, 'optimization_variables.json')
        with open(temp_opt_file, 'w') as f:
            json.dump(optimization_vars, f)
        
        # Create config.json in temp directory
        config = {"num_workers": 1}
        temp_config_file = os.path.join(self.temp_dir, 'config.json')
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)
    
    @patch('main.print_optimization_header')
    @patch('main.create_shared_cache')
    @patch('main.load_optimization_variables')
    @patch('main.load_config')
    @patch('main.print_optimization_parameters')
    @patch('main.EvaluationCallback')
    @patch('main.minimize')
    @patch('main.print_optimization_footer')
    @patch('main.plt.show')
    def test_main_basic_optimization(self, mock_plt_show, mock_footer, mock_minimize,
                                   mock_callback, mock_print_params, mock_load_config,
                                   mock_load_vars, mock_create_cache, mock_header):
        """Test basic main function execution."""
        # Mock return values
        mock_create_cache.return_value = {}
        mock_load_vars.return_value = {"external_wall": ["R3 - Wall"]}
        mock_load_config.return_value = {"num_workers": 1}
        
        # Mock optimization result
        mock_result = Mock()
        mock_result.X = [0]
        mock_result.F = [100.0, 25.0]
        mock_result.algorithm.evaluator.n_eval = 50
        mock_minimize.return_value = mock_result
        
        # Mock callback
        mock_callback_instance = Mock()
        mock_callback.return_value = mock_callback_instance
        
        # Run main function
        main(
            debug_mode=False,
            n_generations=2,
            population_size=4,
            constrained_analysis=False,
            overheating_criterion='ASHRAE',
            test_mode=False
        )
        
        # Check that all expected functions were called
        mock_header.assert_called_once()
        mock_create_cache.assert_called_once()
        mock_load_vars.assert_called_once()
        mock_load_config.assert_called_once()
        mock_print_params.assert_called_once()
        mock_callback.assert_called_once()
        mock_minimize.assert_called_once()
        mock_footer.assert_called_once()
        mock_plt_show.assert_called_once()
    
    @patch('main.print_optimization_header')
    @patch('main.create_shared_cache')
    @patch('main.load_optimization_variables')
    @patch('main.load_config')
    @patch('main.print_optimization_parameters')
    @patch('main.EvaluationCallback')
    @patch('main.minimize')
    @patch('main.print_optimization_footer')
    @patch('main.plt.show')
    def test_main_constrained_optimization(self, mock_plt_show, mock_footer, mock_minimize,
                                         mock_callback, mock_print_params, mock_load_config,
                                         mock_load_vars, mock_create_cache, mock_header):
        """Test main function with constrained optimization."""
        # Mock return values
        mock_create_cache.return_value = {}
        mock_load_vars.return_value = {"external_wall": ["R3 - Wall"]}
        mock_load_config.return_value = {"num_workers": 1}
        
        # Mock optimization result
        mock_result = Mock()
        mock_result.X = [0]
        mock_result.F = [100.0]
        mock_result.algorithm.evaluator.n_eval = 50
        mock_minimize.return_value = mock_result
        
        # Mock callback
        mock_callback_instance = Mock()
        mock_callback.return_value = mock_callback_instance
        
        # Run main function with constrained analysis
        main(
            debug_mode=False,
            n_generations=2,
            population_size=4,
            constrained_analysis=True,
            overheating_criterion='CIBSE59',
            test_mode=False
        )
        
        # Check that constrained optimization was set up correctly
        mock_minimize.assert_called_once()
        call_args = mock_minimize.call_args
        problem = call_args[0][0]  # First argument is the problem
        
        # Should be constrained with CIBSE59
        assert problem.constrained_analysis is True
        assert problem.overheating_criterion == 'CIBSE59'
        assert problem.n_obj == 1  # Single objective in constrained mode
        assert problem.n_ieq_constr == 2  # Two constraints for CIBSE59
        assert problem.test_mode is False  # Test mode should be False
    
    @patch('main.print_optimization_header')
    @patch('main.create_shared_cache')
    @patch('main.load_optimization_variables')
    @patch('main.load_config')
    @patch('main.print_optimization_parameters')
    @patch('main.EvaluationCallback')
    @patch('main.minimize')
    @patch('main.print_optimization_footer')
    @patch('main.plt.show')
    def test_main_test_mode(self, mock_plt_show, mock_footer, mock_minimize,
                           mock_callback, mock_print_params, mock_load_config,
                           mock_load_vars, mock_create_cache, mock_header):
        """Test main function with test mode enabled."""
        # Mock return values
        mock_create_cache.return_value = {}
        mock_load_vars.return_value = {"external_wall": ["R3 - Wall"]}
        mock_load_config.return_value = {"num_workers": 1}
        
        # Mock optimization result
        mock_result = Mock()
        mock_result.X = [0]
        mock_result.F = [100.0, 25.0]
        mock_result.algorithm.evaluator.n_eval = 50
        mock_minimize.return_value = mock_result
        
        # Mock callback
        mock_callback_instance = Mock()
        mock_callback.return_value = mock_callback_instance
        
        # Run main function with test mode enabled
        main(
            debug_mode=False,
            n_generations=2,
            population_size=4,
            constrained_analysis=False,
            overheating_criterion='ASHRAE',
            test_mode=True
        )
        
        # Check that test mode was set up correctly
        mock_minimize.assert_called_once()
        call_args = mock_minimize.call_args
        problem = call_args[0][0]  # First argument is the problem
        
        # Should have test mode enabled
        assert problem.test_mode is True
        assert problem.constrained_analysis is False
        assert problem.overheating_criterion == 'ASHRAE'
    
    @patch('main.print_optimization_header')
    @patch('main.create_shared_cache')
    @patch('main.load_optimization_variables')
    @patch('main.load_config')
    @patch('main.print_optimization_parameters')
    @patch('main.EvaluationCallback')
    @patch('main.minimize')
    @patch('main.print_optimization_footer')
    @patch('main.plt.show')
    def test_main_multiprocessing(self, mock_plt_show, mock_footer, mock_minimize,
                                 mock_callback, mock_print_params, mock_load_config,
                                 mock_load_vars, mock_create_cache, mock_header):
        """Test main function with multiprocessing."""
        # Mock return values
        mock_create_cache.return_value = {}
        mock_load_vars.return_value = {"external_wall": ["R3 - Wall"]}
        mock_load_config.return_value = {"num_workers": 2}  # Multiple workers
        
        # Mock optimization result
        mock_result = Mock()
        mock_result.X = [0]
        mock_result.F = [100.0, 25.0]
        mock_result.algorithm.evaluator.n_eval = 50
        mock_minimize.return_value = mock_result
        
        # Mock callback
        mock_callback_instance = Mock()
        mock_callback.return_value = mock_callback_instance
        
        # Mock multiprocessing pool
        with patch('main.multiprocessing.Pool') as mock_pool_class:
            mock_pool = Mock()
            mock_pool_class.return_value = mock_pool
            
            # Run main function
            main(
                debug_mode=False,
                n_generations=2,
                population_size=4,
                constrained_analysis=False,
                test_mode=False
            )
            
            # Check that multiprocessing was set up
            mock_pool_class.assert_called_once_with(2)
            mock_pool.close.assert_called_once()
            mock_pool.join.assert_called_once()
    
    @patch('main.print_optimization_header')
    @patch('main.create_shared_cache')
    @patch('main.load_optimization_variables')
    @patch('main.load_config')
    @patch('main.print_optimization_parameters')
    @patch('main.EvaluationCallback')
    @patch('main.minimize')
    @patch('main.print_optimization_footer')
    @patch('main.plt.show')
    def test_main_with_custom_parameters(self, mock_plt_show, mock_footer, mock_minimize,
                                       mock_callback, mock_print_params, mock_load_config,
                                       mock_load_vars, mock_create_cache, mock_header):
        """Test main function with custom optimization parameters."""
        # Mock return values
        mock_create_cache.return_value = {}
        mock_load_vars.return_value = {"external_wall": ["R3 - Wall"]}
        mock_load_config.return_value = {"num_workers": 1}
        
        mock_result = Mock()
        mock_result.X = [0]
        mock_result.F = [100.0, 25.0]
        mock_result.algorithm.evaluator.n_eval = 50
        mock_minimize.return_value = mock_result
        
        mock_callback_instance = Mock()
        mock_callback.return_value = mock_callback_instance
        
        # Test with custom parameters
        main(
            debug_mode=True,
            weather_file="custom_weather.epw",
            n_generations=50,
            population_size=100,
            crossover_prob=0.7,
            mutation_prob=0.1,
            seed=12345,
            constrained_analysis=True,
            overheating_criterion='CIBSE59',
            test_mode=False
        )
        
        # Verify custom parameters were used
        mock_minimize.assert_called_once()
        call_args = mock_minimize.call_args
        problem = call_args[0][0]
        
        assert problem.constrained_analysis is True
        assert problem.overheating_criterion == 'CIBSE59'
    
    @patch('main.print_optimization_header')
    @patch('main.create_shared_cache')
    @patch('main.load_optimization_variables')
    @patch('main.load_config')
    def test_main_error_handling(self, mock_load_config, mock_load_vars, mock_create_cache, mock_header):
        """Test main function error handling."""
        # Test with missing optimization variables
        mock_create_cache.return_value = {}
        mock_load_vars.side_effect = FileNotFoundError("File not found")
        
        # Should handle the error gracefully
        with pytest.raises(FileNotFoundError):
            main()
    
    @patch('main.print_optimization_header')
    @patch('main.create_shared_cache')
    @patch('main.load_optimization_variables')
    @patch('main.load_config')
    @patch('main.print_optimization_parameters')
    @patch('main.EvaluationCallback')
    @patch('main.minimize')
    @patch('main.print_optimization_footer')
    @patch('main.plt.show')
    def test_main_seed_generation(self, mock_plt_show, mock_footer, mock_minimize,
                                 mock_callback, mock_print_params, mock_load_config,
                                 mock_load_vars, mock_create_cache, mock_header):
        """Test that seed is generated when not provided."""
        with patch('time.time') as mock_time:  # Fix: patch time.time directly
            
            mock_create_cache.return_value = {}
            mock_load_vars.return_value = {"external_wall": ["R3 - Wall"]}
            mock_load_config.return_value = {"num_workers": 1}
            mock_time.return_value = 1234567890.123  # Fixed time for testing
            
            mock_result = Mock()
            mock_result.X = [0]
            mock_result.F = [100.0, 25.0]
            mock_result.algorithm.evaluator.n_eval = 50
            mock_minimize.return_value = mock_result
            
            mock_callback_instance = Mock()
            mock_callback.return_value = mock_callback_instance
            
            # Run without seed
            main(test_mode=False)
            
            # Verify seed was generated
            mock_minimize.assert_called_once()
            call_args = mock_minimize.call_args
            assert 'seed' in call_args[1]  # seed should be in kwargs


class TestIntegration:
    """Integration tests for the complete system."""
    
    def setup_method(self):
        """Set up test data for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        self.created_files = []
        
        # Create test files in temp directory
        self.create_test_files()
    
    def teardown_method(self):
        """Clean up temporary files."""
        try:
            # Remove any created files first
            for file_path in self.created_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except:
                    pass  # Ignore errors during cleanup
            
            # Remove temporary directory
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
    
    def create_test_files(self):
        """Create test configuration files in temp directory."""
        # Create optimization_variables.json in temp directory
        optimization_vars = {
            "external_wall": ["R3 - Wall", "R9 - Wall"],
            "roof": ["R3 - Roof", "R10 - Roof"],
            "window": ["Dbl LoE", "Trp LoE"]
        }
        filename = os.path.join(self.temp_dir, 'optimization_variables.json')
        with open(filename, 'w') as f:
            json.dump(optimization_vars, f)
        self.created_files.append(filename)
        
        # Create config.json in temp directory
        config = {"num_workers": 1}
        filename = os.path.join(self.temp_dir, 'config.json')
        with open(filename, 'w') as f:
            json.dump(config, f)
        self.created_files.append(filename)
    
    def test_problem_and_callback_integration(self):
        """Test that problem and callback work together correctly."""
        from main import BuildingEnvelopeProblem, EvaluationCallback
        
        # Create shared cache
        shared_cache = {}
        
        # Create problem
        problem = BuildingEnvelopeProblem(
            optimization_variables={"external_wall": ["R3 - Wall", "R9 - Wall"]},
            shared_cache=shared_cache,
            constrained_analysis=False
        )
        
        # Create callback
        with patch('main.initialize_envelope_optimization_plot') as mock_init_plot:
            mock_init_plot.return_value = (Mock(), Mock(), Mock(), Mock())
            
            callback = EvaluationCallback(
                optimization_variables={"external_wall": ["R3 - Wall", "R9 - Wall"]},
                shared_cache=shared_cache,
                constrained_analysis=False
            )
        
        # Verify they're compatible
        assert problem.constrained_analysis == callback.constrained_analysis
        assert problem.overheating_criterion == callback.overheating_criterion
        
        # Verify problem can be evaluated
        x = np.array([0])
        out = {}
        
        # Mock the simulation functions
        with patch('main.run_simulation') as mock_run_sim, \
             patch('main.analyze_simulation_results') as mock_analyze:
            
            mock_run_sim.return_value = "output_dir"
            mock_analyze.return_value = {
                'success': True,
                'district_heating_kwh': 2590.29,
                'average_overheating_hours': 25.0
            }
            
            problem._evaluate(x, out)
            
            # Should have generated objectives
            assert len(out["F"]) == 2
            assert out["F"][0] == 1.0  # TEDI
            assert out["F"][1] == 25.0  # Overheating hours


if __name__ == '__main__':
    pytest.main([__file__])
