import numpy as np
from pymoo.optimize import minimize
import matplotlib.pyplot as plt

from util.simulation_utils import create_shared_cache
from util.cli_utils import (
    get_interactive_arguments,
    print_optimization_header,
    print_optimization_footer,
)
from util.data_utils import (
    load_config,
    print_optimization_parameters,
    print_final_results,
)
from util.pymoo_utils import get_optimization_variables
from config.optimization_config import (
    create_runner,
    create_algorithm,
    EvaluationCallback,
    BuildingEnvelopeProblem,
    cleanup_pool,
)


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
    test_mode=False,
):
    print_optimization_header()

    # Shared cache
    shared_cache = create_shared_cache()

    if debug_mode:
        print("DEBUG MODE: Shared cache will be preserved for inspection")
    if test_mode:
        print("TEST MODE: Using dummy data instead of EnergyPlus simulations")

    # Variables and config
    optimization_variables = get_optimization_variables(optimization_type)
    config = load_config()
    num_workers = config.get('num_workers', 1)
    print(f"Using {num_workers} worker(s) for parallel evaluation.")

    print("Optimization variables loaded:")
    for var_name, var_options in optimization_variables.items():
        try:
            length = len(var_options)
        except Exception:
            length = '?'  # For safety, though our vars are list/dict
        print(f"  {var_name}: {length} options")

    # Runner/pool
    runner, pool = create_runner(num_workers)
    
    print(
        f"Initialized multiprocessing pool with {num_workers} workers" if pool else "Running in single-threaded mode"
    )

    # Problem
    problem = BuildingEnvelopeProblem(
        optimization_variables,
        elementwise_runner=runner,
        shared_cache=shared_cache,
        weather_file=weather_file,
        optimization_type=optimization_type,
        constrained_analysis=constrained_analysis,
        overheating_criterion=overheating_criterion,
        test_mode=test_mode,
    )

    # Seed
    if seed is None:
        import time
        seed = int(time.time() * 1000) % (2**32)
    np.random.seed(seed)

    # Metadata
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
        'Test_Mode': test_mode,
    }
    print_optimization_parameters(metadata)

    # Callback and algorithm
    callback = EvaluationCallback(
        optimization_variables=optimization_variables,
        shared_cache=shared_cache,
        debug_mode=debug_mode,
        metadata=metadata,
        optimization_type=optimization_type,
        constrained_analysis=constrained_analysis,
        overheating_criterion=overheating_criterion,
    )

    method = create_algorithm(optimization_type, population_size, crossover_prob, mutation_prob)

    # Run
    res = minimize(
        problem,
        method,
        termination=('n_gen', n_generations or 100),
        save_history=True,
        callback=callback,
        seed=seed,
    )

    # Cleanup pool
    cleanup_pool(pool)

    # Results
    print_final_results(res, shared_cache)
    callback.save_final_results()
    print_optimization_footer()

    # Hold plot
    print("\nOptimization complete! Plot window will remain open.")
    print("Close the plot window manually when done viewing.")
    plt.ioff()
    plt.show(block=True)


if __name__ == "__main__":
    args = get_interactive_arguments()
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
        test_mode=args['test_mode'],
    )
