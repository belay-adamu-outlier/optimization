import multiprocessing
import numpy as np
from datetime import datetime
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
import os
import matplotlib.pyplot as plt
from energyplus_utils import run_simulation, analyze_simulation_results
from simulation_utils import (
    create_shared_cache,
    make_cache_key,
    get_from_cache,
    set_in_cache,
    get_cache_entry,
    get_generation_from_cache,
    save_cache_to_file
)
from cli_utils import get_interactive_arguments, print_optimization_header, print_optimization_footer
from data_utils import (
    x_to_params,
    print_optimization_table,
    update_pareto_optimality_for_display,
    initialize_csv,
    save_to_csv,
    save_final_results,
    initialize_plot,
    update_plot,
    load_optimization_variables,
    load_config
)


class EvaluationCallback(Callback):
    def __init__(self, optimization_variables, shared_cache=None, debug_mode=False, metadata=None):
        super().__init__()
        self.optimization_variables = optimization_variables
        self.shared_cache = shared_cache
        self.debug_mode = debug_mode
        self.metadata = metadata or {}
        self.generation = 0
        
        # Use dictionaries for efficient lookups instead of lists
        self.solutions_by_key = {}  # combination_key -> solution_dict
        self.solutions_by_generation = {}  # generation -> list of combination_keys
        self.all_objectives = []  # Keep for Pareto comparison (list of [meui, overheating])
        
        # Get variable names for flexible headers
        self.variable_names = list(optimization_variables.keys())
        
        # Initialize CSV file
        self.csv_filename = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        initialize_csv(self.csv_filename, self.variable_names, self.metadata)
        
        # Initialize plotting
        self.fig, self.ax, self.scatter_all, self.scatter_pareto = initialize_plot()
    
    def notify(self, algorithm):
        self.generation += 1
        
        # Get current population
        X = algorithm.pop.get("X")
        F = algorithm.pop.get("F")
        
        # Track results for this generation
        new_evaluations = 0
        skipped_evaluations = 0
        
        for i, (x, f) in enumerate(zip(X, F)):
            # Create parameter dictionary from X
            params = x_to_params(x, self.optimization_variables)
            
            # Create unique key for this combination
            combination_key = make_cache_key(params)
            
            # Check if this combination already exists in cache
            cached_entry = get_cache_entry(self.shared_cache, combination_key)
            
            if cached_entry is not None:
                # This combination has been evaluated before
                cached_generation = get_generation_from_cache(self.shared_cache, combination_key)
                
                if cached_generation is not None and cached_generation < self.generation:
                    # This is a duplicate from a previous generation - SKIP IT
                    skipped_evaluations += 1
                    continue
                elif cached_generation == self.generation:
                    # This was evaluated in current generation - already processed
                    continue
            
            # This is a new combination - add it
            new_evaluations += 1
            
            # Get objective values from F
            meui = f[0]
            overheating = f[1]
            
            # Create solution dictionary
            solution_dict = {
                'generation': self.generation,
                'iteration': i,
                'mechanical_energy_use_intensity': meui,
                'average_overheating_hours': overheating,
                'pareto_optimal': False  # Will be updated for display purposes
            }
            
            # Add all parameters to solution
            for var_name, value in params.items():
                solution_dict[var_name] = value
            
            # Store in our dictionaries
            self.solutions_by_key[combination_key] = solution_dict
            
            # Track by generation
            if self.generation not in self.solutions_by_generation:
                self.solutions_by_generation[self.generation] = []
            self.solutions_by_generation[self.generation].append(combination_key)
            
            # Store objective values for Pareto comparison
            self.all_objectives.append([meui, overheating])
            
            # Update cache with generation info
            if self.shared_cache is not None:
                set_in_cache(self.shared_cache, combination_key, [meui, overheating], self.generation)
        
        # Calculate current Pareto optimality for display purposes
        update_pareto_optimality_for_display(self.solutions_by_key, self.all_objectives)
        
        # Print comprehensive table
        print_optimization_table(self.solutions_by_key, self.variable_names, self.generation)
        
        # Save current generation results to CSV
        save_to_csv(self.csv_filename, self.solutions_by_generation, self.solutions_by_key, self.variable_names, self.generation)
                
        # Print efficiency info
        print(f"\nGeneration {self.generation} Summary:")
        print(f"  Total individuals in generation: {len(X)}")
        print(f"  New unique evaluations: {new_evaluations}")
        print(f"  Duplicate evaluations skipped: {skipped_evaluations}")
        print(f"  Total solutions evaluated so far: {len(self.solutions_by_key)}")
        
        # Update the plot at the end of each generation
        update_plot(self.solutions_by_key, self.generation, self.fig, self.ax, self.scatter_all, self.scatter_pareto)

        # Save cache at the end of each generation if in debug mode
        if self.debug_mode and self.shared_cache is not None:
            save_cache_to_file(self.shared_cache, self.generation)
    
    def save_final_results(self):
        """Save final results with updated Pareto optimality."""
        save_final_results(self.csv_filename, self.solutions_by_key, self.variable_names, self.metadata, self.ax, self.fig)


class BuildingEnvelopeProblem(ElementwiseProblem):
    def __init__(self, optimization_variables, elementwise_runner=None, shared_cache=None, weather_file=None):
        n_var = len(optimization_variables)
        xl = np.zeros(n_var, dtype=int)
        xu = np.zeros(n_var, dtype=int)
        for i, (var_name, var_options) in enumerate(optimization_variables.items()):
            xu[i] = len(var_options) - 1
            
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_ieq_constr=0,
            xl=xl,
            xu=xu,
            vtype=int,
            elementwise_runner=elementwise_runner
        )
        self.optimization_variables = optimization_variables
        self.shared_cache = shared_cache
        self.weather_file = weather_file or "Vancouver - 2021 AMY.epw"

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            
            params = x_to_params(x, self.optimization_variables)
            
            # Create unique key for this combination
            combination_key = make_cache_key(params)
            
            # Check if this combination has already been evaluated using shared cache
            if self.shared_cache is not None:
                cached_result = get_from_cache(self.shared_cache, combination_key)
                if cached_result is not None:
                    out["F"] = cached_result
                    return
            
            # This is a new combination - run simulation
            process_id = os.getpid()
            
            # DUMMY SIMULATION CODE FOR TESTING - REMOVE COMMENTS AND DELETE THIS SECTION WHEN READY
            # import random
            # import time
            # # Simulate some computation time
            # time.sleep(0.1)
            # # Generate dummy results based on parameters
            # # Convert all parameter values to numbers for calculation
            # param_values = []
            # for value in params.values():
            #     if isinstance(value, (int, float)):
            #         param_values.append(value)
            #     elif isinstance(value, str):
            #         # For string values, use their length as a numeric value
            #         param_values.append(len(value))
            #     else:
            #         param_values.append(0)
            
            # meui = random.uniform(50, 200) + sum(param_values) * 10
            # overheating = random.uniform(0, 100) + sum(param_values) * 5
            # objective_values = [meui, overheating]
            # 
            # COMMENT OUT THE ABOVE DUMMY CODE AND UNCOMMENT BELOW FOR REAL SIMULATION
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            output_dir = run_simulation(
                idf_path="Building.idf",
                weather_path=self.weather_file,
                envelope_parameters=params,
                idd_path="C:/EnergyPlusV9-4-0/Energy+.idd",
                output_dir=f"simulation_output_{timestamp}_pid{process_id}",
                verbose='q'
            )
            results = analyze_simulation_results(output_dir)
            if results['success']:
                district_heating = results['district_heating_kwh'] or 0.0
                average_overheating = results['average_overheating_hours'] or 0.0
                mechanical_energy_use_intensity = district_heating / 2590.29
                objective_values = [mechanical_energy_use_intensity, average_overheating]
            else:
                objective_values = [1000.0, 1000.0]
                print(f"Simulation failed: {results.get('error', 'Unknown error')}")
            
            # UNCMMENT UPTO HERE FOR REAL SIMULATION
            
            # Store results in shared cache
            if self.shared_cache is not None:
                set_in_cache(self.shared_cache, combination_key, objective_values)
            
            out["F"] = objective_values
            
        except Exception as e:
            out["F"] = [1000.0, 1000.0]
            print(f"Error evaluating individual: {str(e)}")


def main(debug_mode=False, weather_file=None, n_generations=None, population_size=None, crossover_prob=None, mutation_prob=None):
    # Print optimization header
    print_optimization_header()
    
    # Create shared cache manager
    shared_cache = create_shared_cache()
    
    if debug_mode:
        print("DEBUG MODE: Shared cache will be preserved for inspection")
    
    # Load optimization variables
    optimization_variables = load_optimization_variables()
    
    # Load config
    config = load_config()
    num_workers = config.get('num_workers', 1)
    print(f"Using {num_workers} worker(s) for parallel evaluation.")
    
    # Print optimization variables info
    print("Optimization variables loaded:")
    for var_name, var_options in optimization_variables.items():
        print(f"  {var_name}: {len(var_options)} options")
    
    # Set up parallel processing pool using PyMOO's StarmapParallelization
    if num_workers > 1:
        pool = multiprocessing.Pool(num_workers)
        runner = StarmapParallelization(pool.starmap)
        print(f"Initialized multiprocessing pool with {num_workers} workers")
    else:
        pool = None
        runner = None
        print("Running in single-threaded mode")
    
    # Define problem - pass shared cache to all worker processes
    problem = BuildingEnvelopeProblem(optimization_variables, elementwise_runner=runner, shared_cache=shared_cache, weather_file=weather_file)
    
    # Create metadata for CSV
    metadata = {
        'Algorithm': 'NSGA-II',
        'Weather_File': weather_file or "Vancouver - 2021 AMY.epw",
        'Number_of_Generations': n_generations or 100,
        'Crossover_Probability': crossover_prob or 0.5,
        'Mutation_Probability': mutation_prob or 0.125,
        'Population_Size': population_size or 50,
        'Number_of_Workers': num_workers,
        'Debug_Mode': debug_mode
    }
    
    # Print optimization parameters
    print("\nOptimization Parameters:")
    print(f"  Weather File: {metadata['Weather_File']}")
    print(f"  Number of Generations: {metadata['Number_of_Generations']}")
    print(f"  Crossover Probability: {metadata['Crossover_Probability']}")
    print(f"  Mutation Probability: {metadata['Mutation_Probability']}")
    print(f"  Population Size: {metadata['Population_Size']}")
    print(f"  Algorithm: {metadata['Algorithm']}")
    print(f"  Debug Mode: {metadata['Debug_Mode']}")
    print(f"  Number of Workers: {metadata['Number_of_Workers']}")
    print("-" * 50)
    
    # Define callback with metadata
    callback = EvaluationCallback(optimization_variables, shared_cache=shared_cache, debug_mode=debug_mode, metadata=metadata)
    
    # Define algorithm with built-in duplicate elimination
    method = NSGA2(
        pop_size=population_size or 50,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=crossover_prob or 0.5, eta=5, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=mutation_prob or 0.125, eta=5, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True
    )
    
    # Run optimization
    res = minimize(
        problem,
        method,
        termination=('n_gen', n_generations or 100),
        save_history=True,
        callback=callback
    )
    
    # Clean up pool if used
    if pool:
        pool.close()
        pool.join()
        print("Closed multiprocessing pool")
    
    # Print final cache statistics
    print("\nFinal shared cache statistics:")
    print("  Total cached combinations: {len(shared_cache)}")
    
    # Save final results with updated Pareto optimality
    callback.save_final_results()
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print("Best solution found: %s" % res.X)
    print("Function values: %s" % res.F)
    
    # Print optimization footer
    print_optimization_footer()
    
    # Keep the plot open for viewing
    print("\nOptimization complete! Plot window will remain open.")
    print("Close the plot window manually when done viewing.")
    plt.ioff()  # Turn off interactive mode
    plt.show(block=True)  # Show plot in blocking mode


if __name__ == "__main__":
    # Get arguments interactively
    args = get_interactive_arguments()
    
    # Run main function with parsed arguments
    main(
        debug_mode=args['debug'],
        weather_file=args['weather'],
        n_generations=args['generations'],
        population_size=args['population_size'],
        crossover_prob=args['crossover_prob'],
        mutation_prob=args['mutation_prob']
    ) 