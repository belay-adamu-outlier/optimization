import numpy as np
import json
import multiprocessing
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.core.evaluator import Evaluator
import os


class EvaluationCallback(Callback):
    def __init__(self, constructions_data):
        super().__init__()
        self.constructions_data = constructions_data
        self.generation = 0
        self.all_results = []
        self.all_objectives = []  # Store all objective values for Pareto comparison
        self.unique_combinations = set()  # Track unique construction combinations
        
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def notify(self, algorithm):
        self.generation += 1
        
        # Get current population
        X = algorithm.pop.get("X")
        F = algorithm.pop.get("F")
        
        # Store results for this generation (only unique combinations)
        for i in range(len(X)):
            wall_idx = int(X[i, 0])
            roof_idx = int(X[i, 1])
            window_idx = int(X[i, 2])
            
            wall_name = self.constructions_data['external_walls'][wall_idx]['name']
            roof_name = self.constructions_data['roofs'][roof_idx]['name']
            window_name = self.constructions_data['windows'][window_idx]['name']
            
            energy = F[i, 0]
            cost = F[i, 1]
            
            # Create unique key for this combination
            combination_key = (wall_name, roof_name, window_name)
            
            # Only add if this combination hasn't been seen before
            if combination_key not in self.unique_combinations:
                self.unique_combinations.add(combination_key)
                
                self.all_results.append({
                    'generation': self.generation,
                    'iteration': i,
                    'wall': wall_name,
                    'roof': roof_name,
                    'window': window_name,
                    'energy': energy,
                    'cost': cost,
                    'pareto_optimal': False  # Will be updated after adding to all_objectives
                })
                
                # Store objective values for Pareto comparison
                self.all_objectives.append([energy, cost])
        
        # Now identify Pareto optimal solutions across ALL solutions found so far
        self._update_pareto_optimality()
        
        # Clear screen and print comprehensive table
        self._print_comprehensive_table()
    
    def _print_comprehensive_table(self):
        """Print one comprehensive table with all generations."""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print(f"{'='*120}")
        print(f"COMPREHENSIVE OPTIMIZATION RESULTS - ALL GENERATIONS")
        print(f"{'='*120}")
        
        if not self.all_results:
            return
        
        # Calculate column widths
        wall_width = max(30, max(len(result['wall']) for result in self.all_results))
        roof_width = max(25, max(len(result['roof']) for result in self.all_results))
        window_width = max(40, max(len(result['window']) for result in self.all_results))
        
        # Create header
        header = f"{'Gen':<5} {'Iter':<6} {'Wall':<{wall_width}} {'Roof':<{roof_width}} {'Window':<{window_width}} {'Energy':<15} {'Cost':<15} {'Pareto':<8}"
        print(header)
        print("-" * len(header))
        
        # Print all results
        for result in self.all_results:
            pareto_mark = "✓" if result['pareto_optimal'] else " "
            print(f"{result['generation']:<5} {result['iteration']:<6} {result['wall']:<{wall_width}} {result['roof']:<{roof_width}} {result['window']:<{window_width}} {result['energy']:<15.4f} {result['cost']:<15.2f} {pareto_mark:<8}")
        
        # Print summary
        pareto_count = sum(1 for result in self.all_results if result['pareto_optimal'])
        print(f"Pareto optimal solutions: {pareto_count}")
        print(f"Current generation: {self.generation}")
    
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
    
    def _get_pareto_optimal(self, F):
        """Identify Pareto optimal solutions in the population."""
        n_solutions = F.shape[0]
        pareto_optimal = np.ones(n_solutions, dtype=bool)
        
        for i in range(n_solutions):
            for j in range(n_solutions):
                if i != j:
                    # Check if solution j dominates solution i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        pareto_optimal[i] = False
                        break
        
        return pareto_optimal


class BuildingEnvelopeProblem(Problem):

    def __init__(self, constructions_data):
        super().__init__(
            n_var=3,  # wall, roof, window indices
            n_obj=2,  # energy consumption, cost
            n_ieq_constr=0,
            xl=np.array([0, 0, 0]),  # min indices
            xu=np.array([9, 9, 7]),  # max indices for each type
            vtype=int
        )
        self.constructions_data = constructions_data

    def _evaluate(self, x, out, *args, **kwargs):
        # Extract R-values and calculate objectives
        F = np.zeros((x.shape[0], self.n_obj))
        
        for i in range(x.shape[0]):
            wall_idx = int(x[i, 0])
            roof_idx = int(x[i, 1])
            window_idx = int(x[i, 2])
            
            # Get R-values
            wall_r = self._extract_r_value(self.constructions_data['external_walls'][wall_idx]['name'])
            roof_r = self._extract_r_value(self.constructions_data['roofs'][roof_idx]['name'])
            
            # Objectives
            F[i, 0] = 1.0 / (wall_r + 0.1) + 1.0 / (roof_r + 0.1)  # Energy consumption
            F[i, 1] = wall_r + roof_r  # Construction cost
        
        out["F"] = F
    
    def _extract_r_value(self, construction_name):
        import re
        match = re.search(r'R\s*(\d+(?:\.\d+)?)', construction_name)
        return float(match.group(1)) if match else 1.0


def load_constructions():
    with open('constructions.json', 'r') as f:
        return json.load(f)


def load_config():
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
    # Load construction data
    constructions_data = load_constructions()
    
    # Load config
    config = load_config()
    num_workers = config.get('num_workers', 1)
    print(f"Using {num_workers} worker(s) for parallel evaluation.")
    
    # Define problem
    problem = BuildingEnvelopeProblem(constructions_data)
    
    # Define callback
    callback = EvaluationCallback(constructions_data)
    
    # Define algorithm
    method = NSGA2(
        pop_size=20,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=0.1, eta=20, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True
    )
    
    # Run optimization
    res = minimize(
        problem,
        method,
        termination=('n_gen', 50),
        seed=1,
        save_history=True,
        callback=callback,
        processes=num_workers
    )
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print("Best solution found: %s" % res.X)
    print("Function values: %s" % res.F)


if __name__ == "__main__":
    main() 