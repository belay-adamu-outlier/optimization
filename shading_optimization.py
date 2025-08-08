import multiprocessing
import numpy as np
import os
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

from util.energyplus_utils import run_simulation, analyze_simulation_results
from util.simulation_utils import (
    create_shared_cache,
    make_cache_key,
    get_from_cache,
    set_in_cache,
    save_cache_to_file
)
from util.data_utils import (
    generate_dummy_shading_results,
    print_generation_summary,
    print_optimization_parameters,
    print_final_results,
    print_shading_optimization_table,
    load_config,
    plot_shading_heat_map,
    plot_evolution,
    initialize_plots,
    save_to_csv,
    save_final_results,
    initialize_csv
)


class ShadingOptimizationCallback(Callback):
    def __init__(self, shared_cache=None, debug_mode=False, metadata=None):
        super().__init__()
        self.shared_cache = shared_cache
        self.debug_mode = debug_mode
        self.generation = 0
        self.solutions_by_key = {}
        self.solutions_by_generation = {}
        self.all_objectives = []
        self.metadata = metadata or {}
        
        # CSV setup using generalized initializer
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_filename = 'shading_results_%s.csv' % ts
        initialize_csv(
            self.csv_filename,
            self.metadata,
            problem_type='shading',
            constrained_analysis=False,
            overheating_criterion='CIBSE59'
        )
        
        # Initialize plots
        self.heatmap_fig, self.heatmap_axes, self.evolution_fig, self.evolution_axes = initialize_plots()
        
    def notify(self, algorithm):
        self.generation += 1
        
        # Update the problem's generation tracking
        if hasattr(algorithm.problem, 'set_generation'):
            algorithm.problem.set_generation(self.generation)
        
        # Get current population
        X = algorithm.pop.get("X")
        
        # Calculate new evaluations and skipped evaluations based on cache size change
        current_cache_size = len(self.shared_cache) if self.shared_cache is not None else 0
        new_evaluations = current_cache_size - getattr(self, 'previous_cache_size', 0)
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
                    params = cache_entry.get('params')
                    iteration = i + 1
                    
                    meui, max_lr_oh, max_br_oh = objectives
                    
                    # Create solution dictionary
                    solution_dict = {
                        'generation': generation,
                        'iteration': iteration,
                        'shading_schedule': params.get('shading_schedule', []) if params else [],
                        'mechanical_energy_use_intensity': meui,
                        'max_living_room_overheating': max_lr_oh,
                        'max_bedroom_overheating': max_br_oh
                    }
                    
                    # Store in our dictionaries
                    self.solutions_by_key[combination_key] = solution_dict
                    
                    # Track by generation
                    if generation not in self.solutions_by_generation:
                        self.solutions_by_generation[generation] = []
                    self.solutions_by_generation[generation].append(combination_key)
                    
                    # Store objective values for Pareto comparison
                    self.all_objectives.append([meui, max_lr_oh, max_br_oh])
        
        # Print comprehensive table
        print_shading_optimization_table(self.solutions_by_key)
        
        # Append only unique members of this generation to CSV
        save_to_csv(
            self.csv_filename,
            self.solutions_by_generation,
            self.solutions_by_key,
            [],  # No variable names for shading
            self.generation,
            problem_type='shading',
            constrained_analysis=False,
            overheating_criterion='CIBSE59'
        )

        # Print efficiency info
        print_generation_summary(
            self.generation, 
            total_evaluations_in_generation, 
            new_evaluations, 
            skipped_evaluations, 
            current_cache_size
        )
        
        # Find the best solution in this generation
        if self.solutions_by_key:
            solutions = list(self.solutions_by_key.values())
            # Sort by sum of overheating hours (living room + bedroom)
            solutions.sort(key=lambda x: (x['max_living_room_overheating'] + x['max_bedroom_overheating']))
            best_solution = solutions[0]
            
            # Update plots
            plot_shading_heat_map(
                best_solution['shading_schedule'],
                fig=self.heatmap_fig,
                axes=self.heatmap_axes
            )
            
            # Update evolution plot
            plot_evolution(
                self.solutions_by_key,
                fig=self.evolution_fig,
                axes=self.evolution_axes
            )
        
        # Save cache at the end of each generation if in debug mode
        if self.debug_mode and self.shared_cache is not None:
            save_cache_to_file(self.shared_cache, self.generation)
    
    def finalize_csv(self):
        # Mark Pareto optimal across all generations and rewrite CSV
        save_final_results(
            self.csv_filename, 
            self.solutions_by_key, 
            [],  # No variable names for shading
            self.metadata,
            problem_type='shading',
            constrained_analysis=False,
            overheating_criterion='CIBSE59'
        )
    



class ShadingOptimizationProblem(ElementwiseProblem):
    def __init__(
        self, 
        elementwise_runner=None, 
        shared_cache=None, 
        weather_file=None,
        overheating_criterion='CIBSE59',
        test_mode=False
    ):
        # 144 binary variables for shading optimization (6 hours * 24 hours)
        n_var = 144
        
        # Binary variables: 0 = no shading, 1 = shading
        xl = np.zeros(n_var, dtype=int)  # Lower bound: 0
        xu = np.ones(n_var, dtype=int)   # Upper bound: 1
        
        # No constraints for now - unconstrained optimization
        n_ieq_constr = 0
            
        super().__init__(
            n_var=n_var,
            n_obj=3,  # MEUI, Max Living Room Overheating, Max Bedroom Overheating
            n_ieq_constr=n_ieq_constr,
            xl=xl,
            xu=xu,
            vtype=int,
            elementwise_runner=elementwise_runner
        )
        
        self.shared_cache = shared_cache
        self.weather_file = weather_file or "Vancouver - 2021 AMY.epw"
        self.current_generation = 0  # Track current generation
        self.overheating_criterion = overheating_criterion
        self.test_mode = test_mode

    def set_generation(self, generation):
        """Set the current generation number for tracking purposes."""
        self.current_generation = generation

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            # Convert binary array to shading schedule
            shading_schedule = x.tolist()  # 144-long binary list
            
            # Create unique key for this combination
            combination_key = make_cache_key({'shading_schedule': shading_schedule})
            
            # Check if this combination has already been evaluated using shared cache
            if self.shared_cache is not None:
                cached_result = get_from_cache(self.shared_cache, combination_key)
                if cached_result is not None:
                    objective_values, max_overheating_hours = cached_result
                    out["F"] = objective_values  # Always 2 objectives
                    return
            
            # Use dummy data if in test mode, otherwise run simulation
            if self.test_mode:
                results = generate_dummy_shading_results(shading_schedule)
            else:
                # This is a new combination - run simulation
                process_id = os.getpid()
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                
                # Run simulation with shading schedule
                output_dir = run_simulation(
                    idf_path="Building_Shading.idf",
                    weather_path=self.weather_file,
                    envelope_parameters= {
                        'shading_schedule': shading_schedule
                    },  # Pass shading schedule
                    idd_path="C:/EnergyPlusV9-4-0/Energy+.idd",
                    output_dir=f"shading_simulation_output_{timestamp}_pid{process_id}",
                    verbose='q'
                )
                
                results = analyze_simulation_results(output_dir, overheating_criterion=self.overheating_criterion)
            
            if results['success']:
                district_heating = results['district_heating_kwh'] or 0.0
                mechanical_energy_use_intensity = district_heating / 2590.29
                
                # CIBSE59 overheating analysis
                max_living_room_overheating = results.get('max_living_room_overheating', 0.0) or 0.0
                max_bedroom_overheating = results.get('max_bedroom_overheating', 0.0) or 0.0
                max_overheating_hours = 0.0  # Not used for CIBSE59 but required by set_in_cache
                
                objective_values = [
                    mechanical_energy_use_intensity,
                    max_living_room_overheating,
                    max_bedroom_overheating
                ]
            else:
                objective_values = [1000.0, 1000.0, 1000.0]  # Three objectives
                max_living_room_overheating = 1000.0
                max_bedroom_overheating = 1000.0
                max_overheating_hours = 0.0
                print(f"Simulation failed: {results.get('error', 'Unknown error')}")
            
            # Store results in shared cache
            if self.shared_cache is not None:
                set_in_cache(
                    self.shared_cache, 
                    combination_key, 
                    objective_values, 
                    max_overheating_hours, 
                    generation=self.current_generation,
                    params={'shading_schedule': shading_schedule},
                    max_living_room_overheating=max_living_room_overheating,
                    max_bedroom_overheating=max_bedroom_overheating
                )
            
            out["F"] = objective_values  # Three objectives
            
        except Exception as e:
            out["F"] = [1000.0, 1000.0, 1000.0]  # Three objectives
            print(f"Error evaluating individual: {str(e)}")


def initialize_shading_optimization(
    debug_mode=False, 
    weather_file=None, 
    overheating_criterion='CIBSE59',
    test_mode=False
):
    """Initialize the shading optimization problem and parallel processing setup."""
    
    # Create shared cache manager
    shared_cache = create_shared_cache()
    
    if debug_mode:
        print("DEBUG MODE: Shared cache will be preserved for inspection")
    
    # Load config to get number of workers
    config = load_config()
    num_workers = config.get('num_workers', 1)
    print(f"Using {num_workers} worker(s) for parallel evaluation.")
    
    # Set up parallel processing pool using PyMOO's StarmapParallelization
    if num_workers > 1:
        pool = multiprocessing.Pool(num_workers)
        runner = StarmapParallelization(pool.starmap)
        print(f"Initialized multiprocessing pool with {num_workers} workers")
    else:
        pool = None
        runner = None
        print("Running in single-threaded mode")
    if num_workers > 1:
        pool = multiprocessing.Pool(num_workers)
        runner = StarmapParallelization(pool.starmap)
        print(f"Initialized multiprocessing pool with {num_workers} workers")
    else:
        pool = None
        runner = None
        print("Running in single-threaded mode")
    
    # Define problem - pass shared cache to all worker processes
    problem = ShadingOptimizationProblem(
        elementwise_runner=runner, 
        shared_cache=shared_cache,
        weather_file=weather_file,
        overheating_criterion=overheating_criterion,
        test_mode=test_mode
    )
    
    print("Shading optimization problem initialized:")
    print("  Variables: 144 binary shading decisions")
    print(f"  Weather File: {weather_file or 'Vancouver - 2021 AMY.epw'}")
    print(f"  Overheating Criterion: {overheating_criterion}")
    print(f"  Number of Workers: {num_workers}")
    print(f"  Test Mode: {test_mode}")
    
    return problem, pool, shared_cache


def run_shading_optimization(
    problem,
    pool,
    shared_cache,
    n_generations=50,
    population_size=20,
    crossover_prob=0.8,
    mutation_prob=0.1,
    seed=None
):
    """Run the shading optimization with the specified parameters."""
    
    # Generate seed if not provided
    if seed is None:
        import time
        seed = int(time.time() * 1000) % (2**32)  # Use current time in milliseconds as seed
    
    # Set the seed for reproducibility
    np.random.seed(seed)
    
    # Define algorithm for binary optimization
    
    algorithm = NSGA2(
        pop_size=population_size,
        sampling=BinaryRandomSampling(),
        crossover=UniformCrossover(prob=crossover_prob),
        mutation=BitflipMutation(prob=mutation_prob),
        eliminate_duplicates=True
    )
    
    # Print optimization parameters
    params_dict = {
        'Number of Generations': n_generations,
        'Population Size': population_size,
        'Crossover Probability': crossover_prob,
        'Mutation Probability': mutation_prob,
        'Random Seed': seed
    }
    print_optimization_parameters(params_dict, "NSGA-II with Binary Operators")
    
    # Create callback
    metadata = {
        'Algorithm': 'NSGA-II Binary',
        'Weather_File': problem.weather_file,
        'Generations': n_generations,
        'Population_Size': population_size,
        'Crossover_Probability': crossover_prob,
        'Mutation_Probability': mutation_prob,
        'Seed': seed
    }
    callback = ShadingOptimizationCallback(
        shared_cache=shared_cache,
        debug_mode=False,
        metadata=metadata
    )
    
    # Run optimization
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', n_generations),
        save_history=True,
        seed=seed,
        verbose=True,
        callback=callback
    )
    
    # Print final results
    print_final_results(res, shared_cache)
    # Finalize CSV with Pareto marking
    callback.finalize_csv()
    
    return res


if __name__ == "__main__":
    # Basic initialization test
    problem, pool, shared_cache = initialize_shading_optimization(
        debug_mode=True,
        weather_file="Vancouver - 2021 AMY.epw",
        overheating_criterion='CIBSE59',
        test_mode=True  # Enable test mode for quick testing
    )
    
    print("\nShading optimization problem initialized successfully!")
    
    # Run optimization with default parameters
    try:
        result = run_shading_optimization(
            problem=problem,
            pool=pool,
            shared_cache=shared_cache,
            n_generations=10,  # Small number for testing
            population_size=8,   # Small population for testing
            crossover_prob=0.8,
            mutation_prob=0.1,
            seed=None  # Will generate seed automatically
        )
        
        print("\nOptimization completed successfully!")
        
    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up pool if used
        if pool:
            pool.close()
            pool.join()
            print("Closed multiprocessing pool")
            
        print("\nPlease close the plot windows when you are done reviewing them.")
        plt.show(block=True)