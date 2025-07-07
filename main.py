import numpy as np
import json
import multiprocessing
import csv
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
from energyplus_utils import run_simulation, analyze_simulation_results





class EvaluationCallback(Callback):
    def __init__(self, optimization_variables):
        super().__init__()
        self.optimization_variables = optimization_variables
        self.generation = 0
        self.all_results = []
        self.all_objectives = []  # Store all objective values for Pareto comparison
        self.unique_combinations = set()  # Track unique combinations
        
        # Get variable names for flexible headers
        self.variable_names = list(optimization_variables.keys())
        
        # Initialize CSV file
        self.csv_filename = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._initialize_csv()
        
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def notify(self, algorithm):
        self.generation += 1
        
        # Get current population
        X = algorithm.pop.get("X")
        F = algorithm.pop.get("F")
        
        # Store results for this generation (only new unique combinations)
        new_evaluations = 0
        for i in range(len(X)):
            # Create parameter dictionary from X
            params = self._x_to_params(X[i])
            
            # Create unique key for this combination
            combination_key = tuple(params.values())
            
            # Check if this combination already exists
            existing_result = None
            for result in self.all_results:
                if tuple(result.get(var_name, '') for var_name in self.variable_names) == combination_key:
                    existing_result = result
                    break
            
            if existing_result is None:
                # This is a new combination - add it
                new_evaluations += 1
                result_dict = {
                    'generation': self.generation,
                    'iteration': i,
                    'mechanical_energy_use_intensity': F[i, 0],
                    'average_overheating_hours': F[i, 1],
                    'pareto_optimal': False  # Will be updated after adding to all_objectives
                }
                
                # Add all parameters to result
                for var_name, value in params.items():
                    result_dict[var_name] = value
                
                self.all_results.append(result_dict)
                
                # Store objective values for Pareto comparison
                self.all_objectives.append([F[i, 0], F[i, 1]])
            else:
                # This is a duplicate - just update the generation info
                existing_result['generation'] = self.generation
                existing_result['iteration'] = i
                existing_result['mechanical_energy_use_intensity'] = F[i, 0]
                existing_result['average_overheating_hours'] = F[i, 1]
        
        # Now identify Pareto optimal solutions across ALL solutions found so far
        self._update_pareto_optimality()
        
        # Clear screen and print comprehensive table
        self._print_table()
        
        # Save current generation results to CSV
        self._save_to_csv()
        
        # Print efficiency info
        print(f"\nGeneration {self.generation}: {new_evaluations} new unique evaluations out of {len(X)} total individuals")
    
    def _x_to_params(self, x):
        """Convert optimization variables x to parameter dictionary."""
        params = {}
        var_idx = 0
        
        for var_name, var_options in self.optimization_variables.items():
            if var_name == 'infiltration':
                # Infiltration is a continuous value, not an index
                params[var_name] = var_options[int(x[var_idx])]
            elif var_name.startswith('wwr_'):
                # WWR values are percentages, convert to decimal
                params[var_name] = var_options[int(x[var_idx])] / 100.0
            else:
                # Construction names
                params[var_name] = var_options[int(x[var_idx])]
            var_idx += 1
        
        return params
    
    def _print_table(self):
        """Print Pareto optimal solutions + last 100 iterations."""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print(f"{'='*140}")
        print(f"OPTIMIZATION RESULTS - PARETO OPTIMAL + LAST 100 ITERATIONS")
        print(f"{'='*140}")
        
        if not self.all_results:
            return
        
        # Define fixed column widths with reasonable limits for terminal display
        col_widths = {
            'external_walls': 15,
            'roofs': 15,
            'windows': 15,
            'infiltration': 8,
            'wwr_east': 8,
            'wwr_west': 8,
            'wwr_north': 8,
            'wwr_south': 8
        }
        
        # Create header with fixed widths
        header_parts = [
            f"{'Gen':<4}",
            f"{'Iter':<5}",
        ]
        
        for var_name in self.variable_names:
            if var_name in col_widths:
                header_parts.append(f"{var_name:<{col_widths[var_name]}}")
            else:
                header_parts.append(f"{var_name:<15}")
        
        header_parts.extend([
            f"{'Mech_EUI':<10}",
            f"{'Overheating':<10}",
            f"{'Pareto':<6}"
        ])
        
        header = " ".join(header_parts)
        print(header)
        print("-" * len(header))
        
        # Get Pareto optimal solutions
        pareto_solutions = [result for result in self.all_results if result['pareto_optimal']]
        
        # Get last 100 iterations (or all if less than 100)
        last_100 = self.all_results[-100:] if len(self.all_results) > 100 else self.all_results
        
        # Combine Pareto optimal and last 100, removing duplicates
        solutions_to_show = []
        seen_combinations = set()
        
        # Add Pareto optimal solutions first
        for result in pareto_solutions:
            combination_key = (result['generation'], result['iteration'])
            if combination_key not in seen_combinations:
                solutions_to_show.append(result)
                seen_combinations.add(combination_key)
        
        # Add last 100 iterations (if not already included)
        for result in last_100:
            combination_key = (result['generation'], result['iteration'])
            if combination_key not in seen_combinations:
                solutions_to_show.append(result)
                seen_combinations.add(combination_key)
        
        # Sort by generation and iteration for better readability
        solutions_to_show.sort(key=lambda x: (x['generation'], x['iteration']))
        
        # Print filtered results
        for result in solutions_to_show:
            pareto_mark = "✓" if result['pareto_optimal'] else " "
            
            line_parts = [
                f"{result['generation']:<4}",
                f"{result['iteration']:<5}",
            ]
            
            for var_name in self.variable_names:
                if var_name in result:
                    value = str(result[var_name])
                    # Truncate long values
                    if var_name in col_widths:
                        max_width = col_widths[var_name]
                        if len(value) > max_width:
                            value = value[:max_width-3] + "..."
                        line_parts.append(f"{value:<{max_width}}")
                    else:
                        if len(value) > 15:
                            value = value[:12] + "..."
                        line_parts.append(f"{value:<15}")
                else:
                    if var_name in col_widths:
                        line_parts.append(f"{'':<{col_widths[var_name]}}")
                    else:
                        line_parts.append(f"{'':<15}")
            
            line_parts.extend([
                f"{result['mechanical_energy_use_intensity']:<10.4f}",
                f"{result['average_overheating_hours']:<10.2f}",
                f"{pareto_mark:<6}"
            ])
            
            line = " ".join(line_parts)
            print(line)
        
        # Print summary
        total_solutions = len(self.all_results)
        pareto_count = len(pareto_solutions)
        shown_count = len(solutions_to_show)
        print(f"\nSummary:")
        print(f"  Total solutions evaluated: {total_solutions}")
        print(f"  Pareto optimal solutions: {pareto_count}")
        print(f"  Solutions shown: {shown_count}")
        print(f"  Current generation: {self.generation}")
    
    def _update_pareto_optimality(self):
        """Update Pareto optimality for all solutions found so far."""
        if not self.all_objectives:
            return
            
        all_F = np.array(self.all_objectives)
        n_solutions = len(all_F)
        pareto_optimal = np.ones(n_solutions, dtype=bool)
        
        for i in range(n_solutions):
            for j in range(n_solutions):
                if i != j:
                    # Check if solution j dominates solution i
                    if np.all(all_F[j] <= all_F[i]) and np.any(all_F[j] < all_F[i]):
                        pareto_optimal[i] = False
                        break
        
        # Update the pareto_optimal flag in all_results
        for i, result in enumerate(self.all_results):
            result['pareto_optimal'] = pareto_optimal[i]
    
    def _initialize_csv(self):
        """Initialize the CSV file with headers."""
        with open(self.csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Generation', 'Iteration'] + self.variable_names + ['Mechanical_Energy_Use_Intensity', 'Average_Overheating_Hours', 'Pareto_Optimal']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    def _save_to_csv(self):
        """Save current generation results to CSV file."""
        with open(self.csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Generation', 'Iteration'] + self.variable_names + ['Mechanical_Energy_Use_Intensity', 'Average_Overheating_Hours', 'Pareto_Optimal']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write only the new results from this generation
            for result in self.all_results:
                if result['generation'] == self.generation:
                    row = {
                        'Generation': result['generation'],
                        'Iteration': result['iteration'],
                        'Mechanical_Energy_Use_Intensity': f"{result['mechanical_energy_use_intensity']:.4f}",
                        'Average_Overheating_Hours': f"{result['average_overheating_hours']:.2f}",
                        'Pareto_Optimal': 'Yes' if result['pareto_optimal'] else 'No'
                    }
                    
                    # Add all variable values
                    for var_name in self.variable_names:
                        if var_name in result:
                            row[var_name] = result[var_name]
                    
                    writer.writerow(row)
    
    def save_final_results(self):
        """Save final results with updated Pareto optimality."""
        # Update Pareto optimality one final time
        self._update_pareto_optimality()
        
        # Write all results to CSV with proper Pareto optimality
        with open(self.csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Generation', 'Iteration'] + self.variable_names + ['Mechanical_Energy_Use_Intensity', 'Average_Overheating_Hours', 'Pareto_Optimal']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.all_results:
                row = {
                    'Generation': result['generation'],
                    'Iteration': result['iteration'],
                    'Mechanical_Energy_Use_Intensity': f"{result['mechanical_energy_use_intensity']:.4f}",
                    'Average_Overheating_Hours': f"{result['average_overheating_hours']:.2f}",
                    'Pareto_Optimal': 'Yes' if result['pareto_optimal'] else 'No'
                }
                
                # Add all variable values
                for var_name in self.variable_names:
                    if var_name in result:
                        row[var_name] = result[var_name]
                
                writer.writerow(row)
        
        print(f"\nOptimization complete! Results saved to: {self.csv_filename}")
        print(f"Pareto optimal solutions have been identified and marked in the CSV file.")





class BuildingEnvelopeProblem(ElementwiseProblem):
    def __init__(self, optimization_variables, elementwise_runner=None):
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

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            params = self._x_to_params(x)
            import os
            from datetime import datetime
            process_id = os.getpid()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            output_dir = run_simulation(
                idf_path="Building.idf",
                weather_path="Vancouver - 2021 AMY.epw",
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
                out["F"] = [mechanical_energy_use_intensity, average_overheating]
            else:
                out["F"] = [1000.0, 1000.0]
                print(f"Simulation failed: {results.get('error', 'Unknown error')}")
        except Exception as e:
            out["F"] = [1000.0, 1000.0]
            print(f"Error evaluating individual: {str(e)}")

    def _x_to_params(self, x):
        params = {}
        var_idx = 0
        for var_name, var_options in self.optimization_variables.items():
            if var_name == 'infiltration':
                params[var_name] = var_options[int(x[var_idx])]
            elif var_name.startswith('wwr_'):
                params[var_name] = var_options[int(x[var_idx])] / 100.0
            else:
                params[var_name] = var_options[int(x[var_idx])]
            var_idx += 1
        return params


def load_optimization_variables():
    """Load optimization variables from JSON file."""
    with open('optimization_variables.json', 'r') as f:
        return json.load(f)


def load_config():
    """Load configuration from JSON file."""
    config_path = 'config.json'
    if not os.path.exists(config_path):
        # Create a sample config if it doesn't exist
        sample = {"num_workers": 1}
        with open(config_path, 'w') as f:
            json.dump(sample, f, indent=2)
        return sample
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    # Load optimization variables
    optimization_variables = load_optimization_variables()
    
    # Load config
    config = load_config()
    num_workers = config.get('num_workers', 1)
    print(f"Using {num_workers} worker(s) for parallel evaluation.")
    
    # Print optimization variables info
    print(f"Optimization variables loaded:")
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
    
    # Define problem with parallel runner if available
    if runner:
        problem = BuildingEnvelopeProblem(optimization_variables, elementwise_runner=runner)
    else:
        problem = BuildingEnvelopeProblem(optimization_variables)
    
    # Define callback
    callback = EvaluationCallback(optimization_variables)
    
    # Define algorithm with built-in duplicate elimination
    method = NSGA2(
        pop_size=4,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=0.1, eta=20, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True
    )
    
    # Run optimization
    res = minimize(
        problem,
        method,
        termination=('n_gen', 1),
        seed=1,
        save_history=True,
        callback=callback
    )
    
    # Clean up pool if used
    if pool:
        pool.close()
        pool.join()
        print("Closed multiprocessing pool")
    
    # Save final results with updated Pareto optimality
    callback.save_final_results()
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print("Best solution found: %s" % res.X)
    print("Function values: %s" % res.F)


if __name__ == "__main__":
    main() 