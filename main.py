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
    def __init__(
        self, 
        optimization_variables, 
        shared_cache=None, 
        debug_mode=False, 
        metadata=None,
        constrained_analysis=False
    ):
        super().__init__()
        self.optimization_variables = optimization_variables
        self.shared_cache = shared_cache
        self.debug_mode = debug_mode
        self.metadata = metadata or {}
        self.generation = 0
        self.constrained_analysis = constrained_analysis
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
        self.fig, self.ax, self.scatter_all, self.scatter_pareto = initialize_plot(self.constrained_analysis)
        
        # Track cache size for calculating new evaluations
        self.previous_cache_size = 0
    
    def notify(self, algorithm):
        self.generation += 1
        
        # Update the problem's generation tracking
        if hasattr(algorithm.problem, 'set_generation'):
            algorithm.problem.set_generation(self.generation)
        
        # Get current population
        X = algorithm.pop.get("X")
        
        # Calculate new evaluations and skipped evaluations based on cache size change
        current_cache_size = len(self.shared_cache) if self.shared_cache is not None else 0
        new_evaluations = current_cache_size - self.previous_cache_size
        total_evaluations_in_generation = len(X)
        skipped_evaluations = total_evaluations_in_generation - new_evaluations
        
        # Update previous cache size for next generation
        self.previous_cache_size = current_cache_size
        
        # Build solutions from shared cache (single source of truth)
        self.solutions_by_key = {}
        self.solutions_by_generation = {}
        self.all_objectives = []
        
        # Process all entries in shared cache
        if self.shared_cache is not None:
            for i, (combination_key, cache_entry) in enumerate(self.shared_cache.items()):
                if isinstance(cache_entry, dict) and 'objectives' in cache_entry:
                    objectives = cache_entry['objectives']
                    generation = cache_entry.get('generation', 0)
                    params = cache_entry.get('params') # Get parameters from cache entry
                    iteration = i + 1
                    max_overheating_hours = cache_entry.get('max_overheating_hours')
                
                meui, overheating = objectives
                
                # Create solution dictionary
                solution_dict = {
                    'generation': generation,
                    'iteration': iteration,
                    'mechanical_energy_use_intensity': meui,
                    'average_overheating_hours': overheating,
                    ('max_overheating_hours' if self.constrained_analysis else 'pareto_optimal'): max_overheating_hours if self.constrained_analysis else False
                }
                
                # Add parameters (we need to reconstruct from key)
                if params:
                    for var_name in self.variable_names:
                        solution_dict[var_name] = params.get(var_name, "N/A")
                else:
                    for var_name in self.variable_names:
                        solution_dict[var_name] = "N/A"  # Placeholder
                
                # Store in our dictionaries
                self.solutions_by_key[combination_key] = solution_dict
                
                # Track by generation
                if generation not in self.solutions_by_generation:
                    self.solutions_by_generation[generation] = []
                self.solutions_by_generation[generation].append(combination_key)
                
                # Store objective values for Pareto comparison
                self.all_objectives.append([meui, overheating])
        
        # Calculate current Pareto optimality for display purposes
        if not self.constrained_analysis:
            update_pareto_optimality_for_display(
                self.solutions_by_key, 
                self.all_objectives
            )
        
        # Print comprehensive table
        print_optimization_table(
            self.solutions_by_key, 
            self.variable_names, 
            self.generation,
            self.constrained_analysis
        )
        
        # Save current generation results to CSV
        save_to_csv(
            self.csv_filename, 
            self.solutions_by_generation, 
            self.solutions_by_key, 
            self.variable_names, 
            self.generation
        )
                
        # Print efficiency info
        print(f"\nGeneration {self.generation} Summary:")
        print(f"  Total individuals in generation: {total_evaluations_in_generation}")
        print(f"  New unique evaluations: {new_evaluations}")
        print(f"  Duplicate evaluations skipped: {skipped_evaluations}")
        print(f"  Total unique solutions in cache: {current_cache_size}")
        
        # Update the plot at the end of each generation
        update_plot(
            self.solutions_by_key, 
            self.generation, self.fig, 
            self.ax, self.scatter_all, 
            self.scatter_pareto,
            self.constrained_analysis
        )

        # Save cache at the end of each generation if in debug mode
        if self.debug_mode and self.shared_cache is not None:
            save_cache_to_file(self.shared_cache, self.generation)
    
    def save_final_results(self):
        """Save final results with updated Pareto optimality."""
        save_final_results(
            self.csv_filename, 
            self.solutions_by_key, 
            self.variable_names, 
            self.metadata, self.ax, 
            self.fig,
            self.constrained_analysis
        )


class BuildingEnvelopeProblem(ElementwiseProblem):
    def __init__(
        self, 
        optimization_variables, 
        elementwise_runner=None, 
        shared_cache=None, 
        weather_file=None,
        constrained_analysis=False
    ):
        n_var = len(optimization_variables)
        
        # Set the lower bounds for each variable
        xl = np.zeros(n_var, dtype=int)
        
        # Set the upper bounds for each variable
        xu = np.zeros(n_var, dtype=int)
        for i, (var_name, var_options) in enumerate(optimization_variables.items()):
            xu[i] = len(var_options) - 1
            
        super().__init__(
            n_var=n_var,
            n_obj= 1 if constrained_analysis else 2,
            n_ieq_constr= 1 if constrained_analysis else 0,
            xl=xl,
            xu=xu,
            vtype=int,
            elementwise_runner=elementwise_runner
        )
        
        self.optimization_variables = optimization_variables
        self.shared_cache = shared_cache
        self.weather_file = weather_file or "Vancouver - 2021 AMY.epw"
        self.current_generation = 0  # Track current generation
        self.constrained_analysis = constrained_analysis

    def set_generation(self, generation):
        """Set the current generation number for tracking purposes."""
        self.current_generation = generation

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            
            params = x_to_params(x, self.optimization_variables)
            
            # Create unique key for this combination
            combination_key = make_cache_key(params)
            
            # Check if this combination has already been evaluated using shared cache
            if self.shared_cache is not None:
                cached_result = get_from_cache(self.shared_cache, combination_key)
                if cached_result is not None:
                    objective_values, max_overheating_hours = cached_result
                    out["F"] = objective_values if not self.constrained_analysis else [objective_values[0]]
                    out["G"] = [max_overheating_hours - 30]
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
            # max_overheating_hours = -1
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
                max_overheating_hours = results['max_overheating_hours'] or 0.0
                objective_values = [mechanical_energy_use_intensity, average_overheating]
            else:
                objective_values = [1000.0, 1000.0]
                print(f"Simulation failed: {results.get('error', 'Unknown error')}")
            
            # UNCMMENT UPTO HERE FOR REAL SIMULATION
            
            # Store results in shared cache
            if self.shared_cache is not None:
                set_in_cache(
                    self.shared_cache, 
                    combination_key, 
                    objective_values, 
                    max_overheating_hours, 
                    generation=self.current_generation,
                    params=params
                )
            
            out["F"] = objective_values if not self.constrained_analysis else [objective_values[0]]
            out["G"] = [max_overheating_hours - 30]
            
        except Exception as e:
            out["F"] = [1000.0] if self.constrained_analysis else [1000.0, 1000.0]
            out["G"] = [1000.0]
            print(f"Error evaluating individual: {str(e)}")


def main(
    debug_mode=False, 
    weather_file=None, 
    n_generations=None, 
    population_size=None, 
    crossover_prob=None, 
    mutation_prob=None, 
    seed=None,
    constrained_analysis=False
):
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
    problem = BuildingEnvelopeProblem(
        optimization_variables, 
        elementwise_runner=runner, 
        shared_cache=shared_cache,
        weather_file=weather_file,
        constrained_analysis=constrained_analysis
    )
    
    # Generate seed if not provided
    if seed is None:
        import time
        seed = int(time.time() * 1000) % (2**32)  # Use current time in milliseconds as seed
    
    # Set the seed for reproducibility
    import numpy as np
    np.random.seed(seed)
    
    # Create metadata for CSV
    metadata = {
        'Algorithm': 'NSGA-II',
        'Weather_File': weather_file or "Vancouver - 2021 AMY.epw",
        'Number_of_Generations': n_generations or 100,
        'Crossover_Probability': crossover_prob or 0.5,
        'Mutation_Probability': mutation_prob or 0.125,
        'Population_Size': population_size or 50,
        'Number_of_Workers': num_workers,
        'Debug_Mode': debug_mode,
        'Seed': seed,
        'Constrained_Analysis': constrained_analysis
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
    print(f"  Random Seed: {metadata['Seed']}")
    print(f"  Constrained Analysis: {metadata['Constrained_Analysis']}")
    print("-" * 50)
    
    # Define callback with metadata
    callback = EvaluationCallback(
        optimization_variables, 
        shared_cache=shared_cache, 
        debug_mode=debug_mode, 
        metadata=metadata,
        constrained_analysis=constrained_analysis
    )
    
    # Define algorithm with built-in duplicate elimination and seeding
    method = NSGA2(
        pop_size=population_size or 50,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=crossover_prob or 0.5, eta=5, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=mutation_prob or 0.125, eta=5, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True
    )
    
    # Run optimization with seed
    res = minimize(
        problem,
        method,
        termination=('n_gen', n_generations or 100),
        save_history=True,
        callback=callback,
        seed=seed
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
        mutation_prob=args['mutation_prob'],
        seed=args['seed'],
        constrained_analysis=args['constrained_analysis']
    ) 