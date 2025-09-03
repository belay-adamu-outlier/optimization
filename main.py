import multiprocessing
import numpy as np
from datetime import datetime
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling, FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
import os
import matplotlib.pyplot as plt
from util.energyplus_utils import run_simulation, analyze_simulation_results
from util.simulation_utils import (
    create_shared_cache,
    make_cache_key,
    get_from_cache,
    set_in_cache,
    save_cache_to_file
)
from util.cli_utils import (
    get_interactive_arguments, 
    print_optimization_header, 
    print_optimization_footer
)
from util.data_utils import (
    print_optimization_table,
    update_pareto_optimality_for_display,
    initialize_csv,
    save_to_csv,
    save_final_results,
    initialize_envelope_optimization_plot,
    update_envelope_optimization_plot,
    initialize_shading_optimization_plot,
    update_shading_optimization_plot,
    load_config,
    print_generation_summary,
    print_optimization_parameters,
    print_final_results,
    generate_dummy_envelope_results,
    generate_dummy_shading_results
)
from util.pymoo_utils import (
    get_optimization_variables, 
    get_problem_constraints, 
    get_params,
)


class EvaluationCallback(Callback):
    def __init__(
        self, 
        optimization_variables, 
        shared_cache=None, 
        debug_mode=False, 
        metadata=None,
        optimization_type='envelope',
        constrained_analysis=False,
        overheating_criterion='ASHRAE'
    ):
        super().__init__()
        self.optimization_variables = optimization_variables
        self.shared_cache = shared_cache
        self.debug_mode = debug_mode
        self.metadata = metadata or {}
        self.optimization_type = optimization_type
        self.generation = 0
        self.constrained_analysis = constrained_analysis
        self.overheating_criterion = overheating_criterion
        
        # Set objective names and multiobjective flag for shading optimization
        if optimization_type == 'shading':
            self.objective_names = [
                'Thermal_Energy_Demand_Intensity',
                'Max_Living_Room_Overheating',
                'Max_Bedroom_Overheating'
            ]
            self.multiobjective = True
        else:
            self.objective_names = None
            self.multiobjective = not constrained_analysis
        
        # Use dictionaries for efficient lookups instead of lists
        self.solutions_by_key = {}  # combination_key -> solution_dict
        self.solutions_by_generation = {}  # generation -> list of combination_keys
        self.all_objectives = []  # Keep for Pareto comparison (list of [tedi, overheating])
        
        # Get variable names for flexible headers
        self.variable_names = list(optimization_variables.keys())
        
        # Initialize CSV file
        self.csv_filename = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        initialize_csv(
            self.csv_filename,
            self.metadata,
            variable_names=self.variable_names,
            problem_type=self.optimization_type,
            constrained_analysis=self.constrained_analysis,
            overheating_criterion=self.overheating_criterion
        )
        
        # Initialize plotting based on optimization type
        if self.optimization_type == 'shading':
            self.fig, self.axes, self.scatter_all, self.scatter_pareto = initialize_shading_optimization_plot(
                self.constrained_analysis, 
                self.overheating_criterion
            )
        else:
            self.fig, self.ax, self.scatter_all, self.scatter_pareto = initialize_envelope_optimization_plot(
                self.constrained_analysis, 
                self.overheating_criterion
            )
        
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
                    tedi, overheating = objectives
                
                # Create solution dictionary
                solution_dict = {
                    'generation': generation,
                    'iteration': iteration,
                    'thermal_energy_demand_intensity': tedi,
                    'average_overheating_hours': overheating,
                    ('max_overheating_hours' if self.constrained_analysis else 'pareto_optimal'): max_overheating_hours if self.constrained_analysis else False
                }
                
                # Add CIBSE59 specific values if available
                if self.constrained_analysis and self.overheating_criterion.upper() == 'CIBSE59':
                    solution_dict['max_living_room_overheating'] = cache_entry.get('max_living_room_overheating', 0.0)
                    solution_dict['max_bedroom_overheating'] = cache_entry.get('max_bedroom_overheating', 0.0)
                
                # For shading optimization, always add these values regardless of overheating criterion
                if self.optimization_type == 'shading':
                    solution_dict['max_living_room_overheating'] = cache_entry.get('max_living_room_overheating', 0.0)
                    solution_dict['max_bedroom_overheating'] = cache_entry.get('max_bedroom_overheating', 0.0)
                    solution_dict['total_shading_area'] = cache_entry.get('total_shading_area', 0.0)
                
                # Add optimization variables from params
                if params:
                    for var_name in self.variable_names:
                        if var_name in params:
                            solution_dict[var_name] = params[var_name]
                        else:
                            solution_dict[var_name] = "N/A"
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
                self.all_objectives.append([tedi, overheating])
        
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
            self.constrained_analysis,
            self.overheating_criterion,
            self.optimization_type
        )
        
        # Save all results found so far to CSV (not just current generation)
        save_to_csv(
            self.csv_filename, 
            self.solutions_by_generation, 
            self.solutions_by_key, 
            self.variable_names, 
            None,  # Pass None to save all generations
            problem_type=self.optimization_type,
            constrained_analysis=self.constrained_analysis,
            overheating_criterion=self.overheating_criterion,
            objective_names=self.objective_names,
            multiobjective=self.multiobjective
        )
                
        # Print efficiency info
        print_generation_summary(
            self.generation, 
            total_evaluations_in_generation, 
            new_evaluations, 
            skipped_evaluations, 
            current_cache_size
        )
        
        # Update the plot at the end of each generation
        plot_args = {
            'solutions_by_key': self.solutions_by_key, 
            'generation': self.generation, 
            'fig': self.fig, 
            'axes': self.axes,
            'scatter_all': self.scatter_all,
            'scatter_pareto': self.scatter_pareto,
            'constrained_analysis': self.constrained_analysis, 
            'overheating_criterion': self.overheating_criterion
        }
        
        if self.optimization_type == 'shading':
            update_shading_optimization_plot(**plot_args)
        else:
            update_envelope_optimization_plot(**plot_args)

        # Save cache at the end of each generation if in debug mode
        if self.debug_mode and self.shared_cache is not None:
            save_cache_to_file(self.shared_cache, self.generation)
    
    def save_final_results(self):
        """Save final results with updated Pareto optimality."""
        save_final_results(
            self.csv_filename, 
            self.solutions_by_key, 
            self.variable_names, 
            self.metadata,
            problem_type=self.optimization_type,
            constrained_analysis=self.constrained_analysis,
            overheating_criterion=self.overheating_criterion,
            objective_names=self.objective_names,
            multiobjective=self.multiobjective
        )


class BuildingEnvelopeProblem(ElementwiseProblem):
    def __init__(
        self, 
        optimization_variables, 
        elementwise_runner=None, 
        shared_cache=None, 
        weather_file=None,
        optimization_type='envelope',
        constrained_analysis=False,
        overheating_criterion='ASHRAE',
        test_mode=False
    ):
        n_var = len(optimization_variables)
        
        # Set variable bounds based on optimization type
        if optimization_type == 'shading':
            # For shading optimization, use continuous variables with float bounds
            xl = np.array([bounds['min'] for bounds in optimization_variables.values()], dtype=float)
            xu = np.array([bounds['max'] for bounds in optimization_variables.values()], dtype=float)
            vtype = float
        else:
            # For envelope optimization, use discrete indices
            xl = np.zeros(n_var, dtype=int)
            xu = np.zeros(n_var, dtype=int)
            for i, (var_name, var_options) in enumerate(optimization_variables.items()):
                xu[i] = len(var_options) - 1
            vtype = int
        
        # Determine number of constraints based on optimization type and overheating criterion
        n_ieq_constr = get_problem_constraints(optimization_type, constrained_analysis, overheating_criterion)

        super().__init__(
            n_var=n_var,
            n_obj= 1 if constrained_analysis else 2,
            n_ieq_constr=n_ieq_constr,
            xl=xl,
            xu=xu,
            vtype=vtype,
            elementwise_runner=elementwise_runner
        )
        
        self.optimization_variables = optimization_variables
        self.shared_cache = shared_cache
        self.weather_file = weather_file or "Vancouver - 2021 AMY.epw"
        self.optimization_type = optimization_type
        self.current_generation = 0  # Track current generation
        self.constrained_analysis = constrained_analysis
        self.overheating_criterion = overheating_criterion
        self.test_mode = test_mode

    def set_generation(self, generation):
        """Set the current generation number for tracking purposes."""
        self.current_generation = generation

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            
            # Convert x to params based on optimization type
            params = get_params(self.optimization_type, x, self.optimization_variables)
            # Create unique key for this combination
            combination_key = make_cache_key(params)
            
            # Check if this combination has already been evaluated using shared cache
            if self.shared_cache is not None:
                cached_result = get_from_cache(self.shared_cache, combination_key)
                if cached_result is not None:
                    objective_values, max_overheating_hours = cached_result
                    out["F"] = objective_values if not self.constrained_analysis else [objective_values[0]]
                    if self.constrained_analysis:
                        if self.overheating_criterion.upper() == 'CIBSE59':
                            # For CIBSE59, we need to extract the specific constraint values from cache
                            # The cache should contain max_living_room_overheating and max_bedroom_overheating
                            cache_entry = self.shared_cache[combination_key]
                            if isinstance(cache_entry, dict) and 'max_living_room_overheating' in cache_entry and 'max_bedroom_overheating' in cache_entry:
                                max_living_room_overheating = cache_entry['max_living_room_overheating'] or 0.0
                                max_bedroom_overheating = cache_entry['max_bedroom_overheating'] or 0.0
                                out["G"] = [max_living_room_overheating - 110, max_bedroom_overheating - 32]
                        else:
                            # For ASHRAE, use max_overheating_hours
                            out["G"] = [max_overheating_hours - 30]
                    return
            
            # This is a new combination - run simulation or generate dummy data
            process_id = os.getpid()
            
            if self.test_mode:
                # Use dummy data for testing - just mock the simulation outputs
                if self.optimization_type == 'shading':
                    results = generate_dummy_shading_results(params, self.overheating_criterion)
                else:
                    results = generate_dummy_envelope_results(params, self.overheating_criterion)
            else:
                # REAL SIMULATION CODE - Run actual EnergyPlus simulation
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                
                output_dir = run_simulation(
                    idf_path="Building_SW_Shading_Renamed.idf",
                    weather_path=self.weather_file,
                    params=params,
                    idd_path="C:/EnergyPlusV9-4-0/Energy+.idd",
                    output_dir=f"simulation_output_{timestamp}_pid{process_id}",
                    verbose='q'
                )
                
                results = analyze_simulation_results(
                    output_dir, 
                    overheating_criterion=self.overheating_criterion, 
                    cleanup=True
                )
            
            if results['success']:
                district_heating = results['district_heating_kwh'] or 0.0
                average_overheating = results['average_overheating_hours'] or 0.0
                thermal_energy_demand_intensity = district_heating / 271.24
                max_living_room_overheating = results.get('max_living_room_overheating', 0.0) or 0.0
                max_bedroom_overheating = results.get('max_bedroom_overheating', 0.0) or 0.0
                max_overheating_hours = results.get('max_overheating_hours', 0.0) or 0.0  # Not used for CIBSE59 but required by set_in_cache
                total_shading_area = results.get('total_shading_area', 0.0) or 0.0
                
                objective_values = [thermal_energy_demand_intensity, average_overheating]
            else:
                objective_values = [1000.0, 1000.0]
                max_living_room_overheating = 1000.0
                max_bedroom_overheating = 1000.0
                max_overheating_hours = 0.0  # Not used for CIBSE59 but required by set_in_cache
                total_shading_area = 1000.0
                print(f"Simulation failed: {results.get('error', 'Unknown error')}")
            
            # Store results in shared cache
            if self.shared_cache is not None:
                set_in_cache(
                    self.shared_cache, 
                    combination_key, 
                    objective_values, 
                    max_overheating_hours, 
                    generation=self.current_generation,
                    params=params,
                    max_living_room_overheating=max_living_room_overheating,
                    max_bedroom_overheating=max_bedroom_overheating,
                    total_shading_area=total_shading_area
                )
            
            out["F"] = objective_values if not self.constrained_analysis else [objective_values[0]]
            
            if not self.constrained_analysis:
                return
            
            if self.optimization_type == 'shading':
                    out["G"] = [max_living_room_overheating - 110, max_bedroom_overheating - 32, total_shading_area - 110]
            else:
                if self.overheating_criterion.upper() == 'CIBSE59':
                    out["G"] = [max_living_room_overheating - 110, max_bedroom_overheating - 32]
                else:
                    out["G"] = [max_overheating_hours - 30]
            
        except Exception as e:
            out["F"] = [1000.0] if self.constrained_analysis else [1000.0, 1000.0]
            if self.constrained_analysis:
                if self.optimization_type == 'shading':
                    if self.overheating_criterion.upper() == 'CIBSE59':
                        out["G"] = [1000.0, 1000.0, 1000.0]  # Three constraints for shading CIBSE59
                    else:
                        out["G"] = [1000.0, 1000.0]  # Two constraints for shading ASHRAE
                else:
                    if self.overheating_criterion.upper() == 'CIBSE59':
                        out["G"] = [1000.0, 1000.0]  # Two constraints for envelope CIBSE59
                    else:
                        out["G"] = [1000.0]  # One constraint for envelope ASHRAE
            print(f"Error evaluating individual: {str(e)}")


def main(
    optimization_type='envelope',
    debug_mode=False, 
    weather_file=None, 
    n_generations=None, 
    population_size=None, 
    crossover_prob=None, 
    mutation_prob=None, 
    seed=None,
    constrained_analysis=False,
    overheating_criterion='ASHRAE',
    test_mode=False
):
    # Print optimization header
    print_optimization_header()
    
    # Create shared cache manager
    shared_cache = create_shared_cache()
    
    
    
    if debug_mode:
        print("DEBUG MODE: Shared cache will be preserved for inspection")
    
    if test_mode:
        print("TEST MODE: Using dummy data instead of EnergyPlus simulations")
    
    # Load optimization variables based on optimization type
    optimization_variables = get_optimization_variables(optimization_type)
    
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
        # For single-threaded mode, create a simple runner that processes sequentially
        runner = StarmapParallelization(lambda func, args: [func(*arg) for arg in args])
        print("Running in single-threaded mode")
    
    # Define problem - pass shared cache to all worker processes
    problem = BuildingEnvelopeProblem(
        optimization_variables, 
        elementwise_runner=runner, 
        shared_cache=shared_cache,
        weather_file=weather_file,
        optimization_type=optimization_type,
        constrained_analysis=constrained_analysis,
        overheating_criterion=overheating_criterion,
        test_mode=test_mode
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
        'Optimization_Type': optimization_type,
        'Weather_File': weather_file or "Vancouver - 2021 AMY.epw",
        'Number_of_Generations': n_generations or 100,
        'Crossover_Probability': crossover_prob or 0.5,
        'Mutation_Probability': mutation_prob or 0.125,
        'Population_Size': population_size or 50,
        'Number_of_Workers': num_workers,
        'Debug_Mode': debug_mode,
        'Seed': seed,
        'Constrained_Analysis': constrained_analysis,
        'Overheating_Criterion': overheating_criterion,
        'Test_Mode': test_mode
    }
    
    # Print optimization parameters
    print_optimization_parameters(metadata)
    
    # Define callback with metadata
    callback = EvaluationCallback(
        optimization_variables, 
        shared_cache=shared_cache, 
        debug_mode=debug_mode, 
        metadata=metadata,
        optimization_type=optimization_type,
        constrained_analysis=constrained_analysis,
        overheating_criterion=overheating_criterion
    )
    
    # Define algorithm with built-in duplicate elimination and seeding
    vtype = float if optimization_type == 'shading' else int
    repair = RoundingRepair() if optimization_type == 'envelope' else None
    
    nsga2_args = {
        'pop_size': population_size or 50,
        'sampling': FloatRandomSampling() if optimization_type == 'shading' else IntegerRandomSampling(),
        'crossover': SBX(prob=crossover_prob or 0.5, eta=5, vtype=vtype, repair=repair),
        'mutation': PM(prob=mutation_prob or 0.125, eta=5, vtype=vtype, repair=repair),
        'eliminate_duplicates': True
    }
    
    method = NSGA2(**nsga2_args)
    
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
    
    # Print final results
    print_final_results(res, shared_cache)
    
    # Save final results with updated Pareto optimality
    callback.save_final_results()
    
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
        optimization_type=args['optimization_type'],
        debug_mode=args['debug'],
        weather_file=args['weather'],
        n_generations=args['generations'],
        population_size=args['population_size'],
        crossover_prob=args['crossover_prob'],
        mutation_prob=args['mutation_prob'],
        seed=args['seed'],
        constrained_analysis=args['constrained_analysis'],
        overheating_criterion=args['overheating_criterion'],
        test_mode=args['test_mode']
    ) 
