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
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.core.problem import StarmapParallelization

from util.pymoo_utils import (
    get_problem_constraints,
    get_params,
)
from util.simulation_utils import (
    make_cache_key,
    set_in_cache,
    populate_with_cache,
    compute_shading_penalty,
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
    generate_dummy_data
)
from util.energyplus_utils import (
    run_simulation,
    analyze_simulation_results,
)
import os
from constants.constraints import (
    LR_OVERHEATING_MAX_HRS,
    BR_OVERHEATING_MAX_HRS,
)
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
        if optimization_type in ('shading', 'shading_wwr'):
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
        if self.optimization_type in ('shading', 'shading_wwr'):
            self.fig, self.axes, self.scatter_all, self.scatter_pareto = initialize_shading_optimization_plot(
                self.constrained_analysis, self.overheating_criterion
            )
        else:
            self.fig, self.axes, self.scatter_all, self.scatter_pareto = initialize_envelope_optimization_plot(
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
        self._rebuild_solutions_from_cache()

        if self.multiobjective:
            update_pareto_optimality_for_display(
                self.solutions_by_key,
                self.all_objectives,
                constrained_analysis=self.constrained_analysis,
            )

        print_optimization_table(
            self.solutions_by_key,
            self.variable_names,
            self.generation,
            constrained_analysis=self.constrained_analysis,
            overheating_criterion=self.overheating_criterion,
            problem_type=self.optimization_type,
        )

        if self.optimization_type in ('shading', 'shading_wwr'):
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
                axes=self.axes,
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
            generation=self.generation,
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

    def _rebuild_solutions_from_cache(self):
        """Rebuilds in-memory solution mappings and objectives from the shared cache.

        Populates:
        - self.solutions_by_key: key -> solution summary dict
        - self.solutions_by_generation: generation -> [keys]
        - self.all_objectives: [[tedi, overheating], ...] or 3-objs for shading_wwr
        """
        # Reset containers
        self.solutions_by_key = {}
        self.solutions_by_generation = {}
        self.all_objectives = []

        if not self.shared_cache:
            return

        cibse59 = self.overheating_criterion.upper() == 'CIBSE59'

        for iteration, (combination_key, entry) in enumerate(self.shared_cache.items(), start=1):
            if not isinstance(entry, dict) or 'objectives' not in entry:
                continue

            tedi, overheating = entry['objectives']
            generation = entry.get('generation', 0)
            params = entry.get('params') or {}

            solution = {
                'generation': generation,
                'iteration': iteration,
                'thermal_energy_demand_intensity': tedi,
                'average_overheating_hours': overheating,
            }

            if self.constrained_analysis:
                solution['max_overheating_hours'] = entry.get('max_overheating_hours')
                if cibse59:
                    solution['max_living_room_overheating'] = entry.get('max_living_room_overheating', 0.0)
                    solution['max_bedroom_overheating'] = entry.get('max_bedroom_overheating', 0.0)
                # Feasibility comes from cache; default True if missing
                solution['feasible'] = bool(entry.get('feasible', True))
            else:
                solution['pareto_optimal'] = False
                solution['feasible'] = True

            if self.optimization_type in ('shading', 'shading_wwr'):
                solution['max_living_room_overheating'] = entry.get('max_living_room_overheating', 0.0)
                solution['max_bedroom_overheating'] = entry.get('max_bedroom_overheating', 0.0)
                solution['total_shading_area'] = entry.get('total_shading_area', 0.0)

            for var_name in self.variable_names:
                solution[var_name] = params.get(var_name, "N/A")

            self.solutions_by_key[combination_key] = solution
            self.solutions_by_generation.setdefault(generation, []).append(combination_key)

            if self.optimization_type == 'shading_wwr':
                # 3 objectives: [TEDI_with_penalty, -WWR_south, -WWR_west]
                penalty = compute_shading_penalty(params, self.optimization_variables, self.optimization_type)
                wwr_s = float(params.get('wwr_south', 0.0) or 0.0)
                wwr_w = float(params.get('wwr_west', 0.0) or 0.0)
                solution['tedi_with_penalty'] = float(tedi) + penalty
                self.all_objectives.append([
                    float(tedi) + penalty,
                    -wwr_s,
                    -wwr_w,
                ])
            else:
                self.all_objectives.append([tedi, overheating])

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
        if optimization_type in ('shading', 'shading_wwr'):
            # Integer-encode float variables with 2-dec precision (scale = 100)
            self._precision_scale = 100
            mins = [bounds['min'] for bounds in optimization_variables.values()]
            maxs = [bounds['max'] for bounds in optimization_variables.values()]
            xl = np.rint(np.array(mins, dtype=float) * self._precision_scale).astype(int)
            xu = np.rint(np.array(maxs, dtype=float) * self._precision_scale).astype(int)
            vtype = int
        else:
            xl = np.zeros(n_var, dtype=int)
            xu = np.zeros(n_var, dtype=int)
            for i, (var_name, var_options) in enumerate(optimization_variables.items()):
                xu[i] = len(var_options) - 1
            vtype = int

        n_ieq_constr = get_problem_constraints(optimization_type, constrained_analysis, overheating_criterion)

        # Determine number of objectives
        if optimization_type == 'shading_wwr':
            # Min TEDI (with penalty), Max WWR_south, Max WWR_west -> 3 objectives
            n_obj = 3
        else:
            # Envelope: 2 when unconstrained, 1 when constrained
            n_obj = 1 if constrained_analysis else 2

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=n_ieq_constr,
            xl=xl,
            xu=xu,
            vtype=vtype,
            elementwise_runner=elementwise_runner,
        )

        self.optimization_variables = optimization_variables
        self.shared_cache = shared_cache
        self.weather_file = weather_file or "BEMs/Vancouver - 2021 AMY.epw"
        self.optimization_type = optimization_type
        self.current_generation = 0
        self.constrained_analysis = constrained_analysis
        self.overheating_criterion = overheating_criterion
        self.test_mode = test_mode

    def set_generation(self, generation):
        self.current_generation = generation


    def _evaluate(self, x, out, *args, **kwargs):
        try:
            # Decode integer-encoded shading variables back to 2-dec floats for params
            if self.optimization_type in ('shading', 'shading_wwr'):
                x_decoded = (np.asarray(x, dtype=float) / float(getattr(self, '_precision_scale', 1))).tolist()
            else:
                x_decoded = x

            params = get_params(self.optimization_type, x_decoded, self.optimization_variables)
            combination_key = make_cache_key(params)

            # Flag for shading_wwr; populate_with_cache now returns 3 objectives already
            is_shading_wwr = (self.optimization_type == 'shading_wwr')

            # Try cache first; if hit, populate out with penalized TEDI (cache remains raw)
            if populate_with_cache(
                self.shared_cache,
                combination_key,
                self.constrained_analysis,
                out,
                optimization_type=self.optimization_type,
                optimization_variables=self.optimization_variables,
            ):
                return

            process_id = os.getpid()

            if self.test_mode:
                results = generate_dummy_data(params, self.optimization_type, self.overheating_criterion)
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]

                params['wwr_east'] = 0.2
                params['wwr_west'] = 0.2
                params['wwr_north'] = 0.22
                params['wwr_south'] = 0.2
                params['window'] = 'Dbl LoE Elec Ref Colored 6mm/13m Arg - 1001'
                params['external_wall'] = 'R41 - Wall'
                params['roof'] = 'R18 - Roof'
                params['infiltration_ach'] = 0.78
                params['wall_solar_absorptance'] = 0.15
                params['roof_solar_absorptance'] = 0.15
    
                
                output_dir = run_simulation(
                    idf_path="BEMs/Building.idf",
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
                tedi = district_heating / 2590.29
                lr = results.get('max_living_room_overheating', 0.0) or 0.0
                br = results.get('max_bedroom_overheating', 0.0) or 0.0
                max_overheating_hours = results.get('max_overheating_hours', 0.0) or 0.0
                total_area = results.get('total_shading_area', 0.0) or 0.0
                # Store raw objectives in cache; apply penalty now for optimizer
                objectives_raw = [tedi, overheating]
                penalty = compute_shading_penalty(params, self.optimization_variables, self.optimization_type)
                tedi_with_penalty = tedi + penalty
            else:
                tedi = 1000.0
                overheating = 1000.0
                objectives_raw = [tedi, overheating]
                tedi_with_penalty = tedi
                lr, br, max_overheating_hours, total_area = 1000.0, 1000.0, 0.0, 1000.0

            if self.shared_cache is not None:
                feasible = self._is_feasible(lr, br) if self.constrained_analysis else True
                set_in_cache(
                    self.shared_cache,
                    combination_key,
                    objectives_raw,
                    max_overheating_hours,
                    generation=self.current_generation,
                    params=params,
                    max_living_room_overheating=lr,
                    max_bedroom_overheating=br,
                    total_shading_area=total_area,
                    feasible=feasible,
                )

            # Set outputs directly
            if is_shading_wwr:
                # Build WWR objectives here (no penalty on WWRs)
                f_wwr_s = -float(params.get('wwr_south') or 0.0)
                f_wwr_w = -float(params.get('wwr_west') or 0.0)
                out['F'] = [tedi_with_penalty, f_wwr_s, f_wwr_w]
                if self.constrained_analysis:
                    out['G'] = [lr - LR_OVERHEATING_MAX_HRS, br - BR_OVERHEATING_MAX_HRS]
                return

            if not self.constrained_analysis:
                out["F"] = [tedi_with_penalty, overheating]
                return
            out["F"] = [tedi_with_penalty]
            out["G"] = [lr - LR_OVERHEATING_MAX_HRS, br - BR_OVERHEATING_MAX_HRS]
            return
            
        except Exception as _:
            if self.optimization_type == 'shading_wwr':
                out['F'] = [1000.0, 1000.0, 1000.0]
            else:
                out['F'] = [1000.0] if self.constrained_analysis else [1000.0, 1000.0]
            if not self.constrained_analysis:
                return
            out["G"] = [1000.0, 1000.0]

    def _is_feasible(self, lr: float, br: float) -> bool:
        """Return True if solution respects overheating constraints.

        Defaults to True on error to avoid discarding results.
        """
        try:
            lr_v = float(lr or 0.0)
            br_v = float(br or 0.0)
            return (lr_v <= LR_OVERHEATING_MAX_HRS) and (br_v <= BR_OVERHEATING_MAX_HRS)
        except Exception:
            return True

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

def create_algorithm(
    optimization_type, 
    population_size=None, 
    crossover_prob=None, 
    mutation_prob=None
) -> NSGA2:
    # Envelope uses integer categories; shading now uses integer-encoded floats (scale=100)
    repair = RoundingRepair() if optimization_type == 'envelope' else None

    args = {
        'pop_size': population_size or 50,
        'sampling': IntegerRandomSampling(),
        'crossover': SBX(prob=crossover_prob or 0.5, eta=5, vtype=int, repair=repair),
        'mutation': PM(prob=mutation_prob or 0.125, eta=5, vtype=int, repair=repair),
        'eliminate_duplicates': True,
    }

    return NSGA2(**args)
