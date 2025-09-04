"""
Optimization configuration module.

Contains the Problem and Callback classes, and small helpers to
assemble the runner and the NSGA-II algorithm so main.py can focus
on initialization, orchestration, and I/O.
"""

import numpy as np
from datetime import datetime
from typing import Tuple, Optional

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling, FloatRandomSampling
from pymoo.core.problem import StarmapParallelization

from util.pymoo_utils import (
    get_problem_constraints,
    get_params,
)
from util.simulation_utils import (
    make_cache_key,
    get_from_cache,
    set_in_cache,
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
    print_generation_summary,
)
from util.energyplus_utils import (
    run_simulation,
    analyze_simulation_results,
)
from util.data_utils import (
    generate_dummy_envelope_results,
    generate_dummy_shading_results,
)
import os


class EvaluationCallback(Callback):
    def __init__(
        self,
        optimization_variables,
        shared_cache=None,
        debug_mode: bool = False,
        metadata: Optional[dict] = None,
        optimization_type: str = 'envelope',
        constrained_analysis: bool = False,
        overheating_criterion: str = 'ASHRAE',
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
                'Max_Bedroom_Overheating',
            ]
            self.multiobjective = True
        else:
            self.objective_names = None
            self.multiobjective = not constrained_analysis

        # Use dictionaries for efficient lookups instead of lists
        self.solutions_by_key = {}
        self.solutions_by_generation = {}
        self.all_objectives = []

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
            overheating_criterion=self.overheating_criterion,
        )

        # Initialize plotting based on optimization type
        if self.optimization_type == 'shading':
            self.fig, self.axes, self.scatter_all, self.scatter_pareto = initialize_shading_optimization_plot(
                self.constrained_analysis, self.overheating_criterion
            )
        else:
            self.fig, self.ax, self.scatter_all, self.scatter_pareto = initialize_envelope_optimization_plot(
                self.constrained_analysis, self.overheating_criterion
            )

        # Track cache size for calculating new evaluations
        self.previous_cache_size = 0

    def notify(self, algorithm):
        self.generation += 1

        # Update the problem's generation tracking
        if hasattr(algorithm.problem, 'set_generation'):
            algorithm.problem.set_generation(self.generation)

        # Current population size and new/skipped eval estimates can be derived
        X = algorithm.pop.get("X")
        current_cache_size = len(self.shared_cache) if self.shared_cache is not None else 0
        new_evaluations = current_cache_size - self.previous_cache_size
        skipped_evaluations = len(X) - new_evaluations
        self.previous_cache_size = current_cache_size

        # Rebuild solutions listing from cache for plotting/reporting
        self.solutions_by_key = {}
        self.solutions_by_generation = {}
        self.all_objectives = []

        if self.shared_cache is not None:
            for i, (combination_key, cache_entry) in enumerate(self.shared_cache.items()):
                if isinstance(cache_entry, dict) and 'objectives' in cache_entry:
                    objectives = cache_entry['objectives']
                    generation = cache_entry.get('generation', 0)
                    params = cache_entry.get('params')
                    iteration = i + 1
                    max_overheating_hours = cache_entry.get('max_overheating_hours')
                    tedi, overheating = objectives

                    solution_dict = {
                        'generation': generation,
                        'iteration': iteration,
                        'thermal_energy_demand_intensity': tedi,
                        'average_overheating_hours': overheating,
                        (
                            'max_overheating_hours' if self.constrained_analysis else 'pareto_optimal'
                        ): max_overheating_hours if self.constrained_analysis else False,
                    }

                    if self.constrained_analysis and self.overheating_criterion.upper() == 'CIBSE59':
                        solution_dict['max_living_room_overheating'] = cache_entry.get(
                            'max_living_room_overheating', 0.0
                        )
                        solution_dict['max_bedroom_overheating'] = cache_entry.get(
                            'max_bedroom_overheating', 0.0
                        )

                    if self.optimization_type == 'shading':
                        solution_dict['max_living_room_overheating'] = cache_entry.get(
                            'max_living_room_overheating', 0.0
                        )
                        solution_dict['max_bedroom_overheating'] = cache_entry.get(
                            'max_bedroom_overheating', 0.0
                        )
                        solution_dict['total_shading_area'] = cache_entry.get('total_shading_area', 0.0)

                    if params:
                        for var_name in self.variable_names:
                            solution_dict[var_name] = params.get(var_name, "N/A")
                    else:
                        for var_name in self.variable_names:
                            solution_dict[var_name] = "N/A"

                    self.solutions_by_key[combination_key] = solution_dict
                    if generation not in self.solutions_by_generation:
                        self.solutions_by_generation[generation] = []
                    self.solutions_by_generation[generation].append(combination_key)
                    self.all_objectives.append([tedi, overheating])

        if not self.constrained_analysis:
            update_pareto_optimality_for_display(self.solutions_by_key, self.all_objectives)

        print_optimization_table(
            self.solutions_by_key,
            self.variable_names,
            self.generation,
            constrained_analysis=self.constrained_analysis,
            overheating_criterion=self.overheating_criterion,
            problem_type=self.optimization_type,
        )

        if self.optimization_type == 'shading':
            update_shading_optimization_plot(
                self.solutions_by_key,
                generation=self.generation,
                axes=self.axes,
                fig=self.fig,
                scatter_all=self.scatter_all,
                scatter_pareto=self.scatter_pareto,
                constrained_analysis=self.constrained_analysis,
                overheating_criterion=self.overheating_criterion,
            )
        else:
            update_envelope_optimization_plot(
                self.solutions_by_key,
                generation=self.generation,
                fig=self.fig,
                ax=self.ax,
                scatter_all=self.scatter_all,
                scatter_pareto=self.scatter_pareto,
                constrained_analysis=self.constrained_analysis,
                overheating_criterion=self.overheating_criterion,
            )

        save_to_csv(
            self.csv_filename,
            self.solutions_by_generation,
            self.solutions_by_key,
            self.variable_names,
            generation=None,
            problem_type=self.optimization_type,
            constrained_analysis=self.constrained_analysis,
            overheating_criterion=self.overheating_criterion,
            objective_names=self.objective_names,
            multiobjective=self.multiobjective,
        )
        # Show generation summary consistent with data_utils API
        try:
            total_evals = len(X)
        except Exception:
            total_evals = 0
        print_generation_summary(
            self.generation,
            total_evals,
            new_evaluations,
            skipped_evaluations,
            current_cache_size,
        )

    def save_final_results(self):
        save_final_results(
            self.csv_filename,
            self.solutions_by_key,
            variable_names=self.variable_names,
            metadata=self.metadata,
            problem_type=self.optimization_type,
            constrained_analysis=self.constrained_analysis,
            overheating_criterion=self.overheating_criterion,
            objective_names=self.objective_names,
            multiobjective=self.multiobjective,
        )


class BuildingEnvelopeProblem(ElementwiseProblem):
    def __init__(
        self,
        optimization_variables,
        elementwise_runner=None,
        shared_cache=None,
        weather_file=None,
        optimization_type: str = 'envelope',
        constrained_analysis: bool = False,
        overheating_criterion: str = 'ASHRAE',
        test_mode: bool = False,
    ):
        n_var = len(optimization_variables)

        # Set variable bounds based on optimization type
        if optimization_type == 'shading':
            xl = np.array([bounds['min'] for bounds in optimization_variables.values()], dtype=float)
            xu = np.array([bounds['max'] for bounds in optimization_variables.values()], dtype=float)
            vtype = float
        else:
            xl = np.zeros(n_var, dtype=int)
            xu = np.zeros(n_var, dtype=int)
            for i, (var_name, var_options) in enumerate(optimization_variables.items()):
                xu[i] = len(var_options) - 1
            vtype = int

        n_ieq_constr = get_problem_constraints(optimization_type, constrained_analysis, overheating_criterion)

        super().__init__(
            n_var=n_var,
            n_obj=1 if constrained_analysis else 2,
            n_ieq_constr=n_ieq_constr,
            xl=xl,
            xu=xu,
            vtype=vtype,
            elementwise_runner=elementwise_runner,
        )

        self.optimization_variables = optimization_variables
        self.shared_cache = shared_cache
        self.weather_file = weather_file or "Vancouver - 2021 AMY.epw"
        self.optimization_type = optimization_type
        self.current_generation = 0
        self.constrained_analysis = constrained_analysis
        self.overheating_criterion = overheating_criterion
        self.test_mode = test_mode

    def set_generation(self, generation):
        self.current_generation = generation

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            params = get_params(self.optimization_type, x, self.optimization_variables)
            combination_key = make_cache_key(params)

            if self.shared_cache is not None:
                cached_result = get_from_cache(self.shared_cache, combination_key)
                if cached_result is not None:
                    objectives, max_overheating_hours = cached_result
                    out["F"] = objectives if not self.constrained_analysis else [objectives[0]]
                    if self.constrained_analysis:
                        if self.overheating_criterion.upper() == 'CIBSE59':
                            cache_entry = self.shared_cache[combination_key]
                            if (
                                isinstance(cache_entry, dict)
                                and 'max_living_room_overheating' in cache_entry
                                and 'max_bedroom_overheating' in cache_entry
                            ):
                                lr = cache_entry['max_living_room_overheating'] or 0.0
                                br = cache_entry['max_bedroom_overheating'] or 0.0
                                out["G"] = [lr - 110, br - 32]
                        else:
                            out["G"] = [max_overheating_hours - 30]
                    return

            process_id = os.getpid()

            if self.test_mode:
                if self.optimization_type == 'shading':
                    results = generate_dummy_shading_results(params, self.overheating_criterion)
                else:
                    results = generate_dummy_envelope_results(params, self.overheating_criterion)
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                output_dir = run_simulation(
                    idf_path="Building_SW_Shading_Renamed.idf",
                    weather_path=self.weather_file,
                    params=params,
                    idd_path="C:/EnergyPlusV9-4-0/Energy+.idd",
                    output_dir=f"simulation_output_{timestamp}_pid{process_id}",
                    verbose='q',
                )
                results = analyze_simulation_results(
                    output_dir,
                    overheating_criterion=self.overheating_criterion,
                    cleanup=True,
                )

            if results['success']:
                district_heating = results['district_heating_kwh'] or 0.0
                overheating = results['average_overheating_hours'] or 0.0
                tedi = district_heating / 271.24
                lr = results.get('max_living_room_overheating', 0.0) or 0.0
                br = results.get('max_bedroom_overheating', 0.0) or 0.0
                max_overheating_hours = results.get('max_overheating_hours', 0.0) or 0.0
                total_area = results.get('total_shading_area', 0.0) or 0.0
                objectives = [tedi, overheating]
            else:
                objectives = [1000.0, 1000.0]
                lr, br, max_overheating_hours, total_area = 1000.0, 1000.0, 0.0, 1000.0

            if self.shared_cache is not None:
                set_in_cache(
                    self.shared_cache,
                    combination_key,
                    objectives,
                    max_overheating_hours,
                    generation=self.current_generation,
                    params=params,
                    max_living_room_overheating=lr,
                    max_bedroom_overheating=br,
                    total_shading_area=total_area,
                )

            out["F"] = objectives if not self.constrained_analysis else [objectives[0]]
            if not self.constrained_analysis:
                return
            if self.optimization_type == 'shading':
                out["G"] = [lr - 110, br - 32, total_area - 110]
            else:
                if self.overheating_criterion.upper() == 'CIBSE59':
                    out["G"] = [lr - 110, br - 32]
                else:
                    out["G"] = [max_overheating_hours - 30]
        except Exception as e:
            out["F"] = [1000.0] if self.constrained_analysis else [1000.0, 1000.0]
            if self.constrained_analysis:
                if self.optimization_type == 'shading':
                    if self.overheating_criterion.upper() == 'CIBSE59':
                        out["G"] = [1000.0, 1000.0, 1000.0]
                    else:
                        out["G"] = [1000.0, 1000.0]
                else:
                    if self.overheating_criterion.upper() == 'CIBSE59':
                        out["G"] = [1000.0, 1000.0]
                    else:
                        out["G"] = [1000.0]


def create_runner(num_workers: int) -> Tuple[StarmapParallelization, Optional[object]]:
    if num_workers > 1:
        import multiprocessing
        pool = multiprocessing.Pool(num_workers)
        runner = StarmapParallelization(pool.starmap)
        return runner, pool
    else:
        runner = StarmapParallelization(lambda func, args: [func(*arg) for arg in args])
        return runner, None

def cleanup_pool(pool):
    if pool:
        pool.close()
        pool.join()
        print("Closed multiprocessing pool")


def create_algorithm(optimization_type, population_size=None, crossover_prob=None, mutation_prob=None) -> NSGA2:
    vtype = float if optimization_type == 'shading' else int
    repair = RoundingRepair() if optimization_type == 'envelope' else None
    
    args = {
        'pop_size': population_size or 50,
        'sampling': FloatRandomSampling() if optimization_type == 'shading' else IntegerRandomSampling(),
        'crossover': SBX(prob=crossover_prob or 0.5, eta=5, vtype=vtype, repair=repair),
        'mutation': PM(prob=mutation_prob or 0.125, eta=5, vtype=vtype, repair=repair),
        'eliminate_duplicates': True,
    }
    
    return NSGA2(**args)
