"""
Data Utilities for Building Envelope Optimization

This module contains utility functions for:
- Parameter conversion
- Table printing and formatting
- Pareto optimality calculations
- CSV file operations
- Plot management
- Configuration loading
"""

import json
import os
import csv
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.ion()  # Enable interactive mode globally

def x_to_params(x, optimization_variables):
    """Convert optimization variables x to parameter dictionary."""
    params = {}
    var_idx = 0
    
    for var_name, var_options in optimization_variables.items():
        if var_name.startswith('wwr_'):
            # WWR values are percentages, convert to decimal
            params[var_name] = var_options[int(x[var_idx])] / 100.0
        else:
            # Construction names
            params[var_name] = var_options[int(x[var_idx])]
        var_idx += 1
    
    return params


def print_optimization_table(
    solutions_by_key, 
    variable_names, 
    generation, 
    constrained_analysis=False,
    overheating_criterion='ASHRAE'
):

    """Print a formatted table of optimization results."""
    if not solutions_by_key:
        print("No solutions to display yet.")
        return
        
    # Calculate column widths for variable names
    col_widths = {}
    for var_name in variable_names:
        # Start with the variable name length
        max_width = len(var_name)
        # Check all values for this variable
        for result in solutions_by_key.values():
            if var_name in result:
                value_str = str(result[var_name])
                max_width = max(max_width, len(value_str))
        # Add some padding and cap at reasonable length
        col_widths[var_name] = min(max_width + 2, 15)
    
    print(f"\n{'='*100}")
    print(f"OPTIMIZATION RESULTS - {'PARETO OPTIMAL + ' if not constrained_analysis else ''} LAST 100 ITERATIONS")
    print(f"{'='*100}")
    
    # Create header with fixed widths
    header_parts = [
        f"{'Gen':<3}",
        f"{'Iter':<4}",
    ]
    
    for var_name in variable_names:
        if var_name in col_widths:
            name = var_name
            max_width = col_widths[var_name]
            if len(name) > max_width:
                name = name[:max_width-3] + "..."
            header_parts.append(f"{name:<{max_width}}")
        else:
            name = var_name
            max_width = 8
            if len(name) > max_width:
                name = name[:5] + "..."
            header_parts.append(f"{name:<{max_width}}")
    
    if constrained_analysis and overheating_criterion.upper() == 'CIBSE59':
        header_parts.extend([
            f"{'Mech_EUI':<8}",
            f"{'Max_LR_OH':<12}",
            f"{'Max_BR_OH':<12}"
        ])
    else:
        header_parts.extend([
            f"{'Mech_EUI':<8}",
            f"{'Overheating':<8}",
            f"{'Pareto_Opt':<10}" if not constrained_analysis else f"{'Max_Overheating':<8}"
        ])
    
    header = " ".join(header_parts)
    print(header)
    print("-" * len(header))
    
    # Get all solutions as a list for processing
    all_solutions = list(solutions_by_key.values())
    
    # Get Pareto optimal solutions
    pareto_solutions = [result for result in all_solutions if result['pareto_optimal']] if not constrained_analysis else []
    
    # Get last 100 iterations (or all if less than 100)
    last_100 = all_solutions[-100:] if len(all_solutions) > 100 else all_solutions
    
    # Combine Pareto optimal and last 100, removing duplicates
    solutions_to_show = []
    
    # Add Pareto optimal solutions first
    for result in pareto_solutions: 
        solutions_to_show.append(result)
    
    # Add last 100 iterations (if not already included)
    for result in last_100:
        solutions_to_show.append(result)
    
    # Sort by generation and iteration for better readability
    solutions_to_show.sort(key=lambda x: (x['generation'], x['iteration']))
    
    # Print filtered results
    for result in solutions_to_show:
        if not constrained_analysis:
            pareto_mark = "✓" if result['pareto_optimal'] else " "
        
        line_parts = [
            f"{result.get('generation') or 0:<3}",
            f"{result.get('iteration') or 0:<4}",
        ]
        
        for var_name in variable_names:
            if var_name in result:
                value = str(result[var_name])
                # Truncate long values and ensure proper padding
                if var_name in col_widths:
                    max_width = col_widths[var_name]
                    if len(value) > max_width:
                        value = value[:max_width-3] + "..."
                    line_parts.append(f"{value:<{max_width}}")
                else:
                    if len(value) > 8:
                        value = value[:5] + "..."
                    line_parts.append(f"{value:<8}")
            else:
                # Empty value - pad to the appropriate width
                if var_name in col_widths:
                    line_parts.append(f"{'':<{col_widths[var_name]}}")
                else:
                    line_parts.append(f"{'':<8}")
        
        if constrained_analysis and overheating_criterion.upper() == 'CIBSE59':
            line_parts.extend([
                f"{result['mechanical_energy_use_intensity']:<8.3f}",
                f"{result.get('max_living_room_overheating', 0.0):<12.1f}",
                f"{result.get('max_bedroom_overheating', 0.0):<12.1f}"
            ])
        else:
            line_parts.extend([
                f"{result['mechanical_energy_use_intensity']:<8.3f}",
                f"{result['average_overheating_hours']:<8.1f}",
                f"{pareto_mark:<10} " if not constrained_analysis else f"{result['max_overheating_hours']:<8.1f}"
            ])
        
        line = " ".join(line_parts)
        print(line)
    
    # Print summary
    total_solutions = len(solutions_by_key)
    pareto_count = len(pareto_solutions)
    shown_count = len(solutions_to_show)
    print("\nSummary:")
    print(f"  Total solutions evaluated: {total_solutions}")
    if not constrained_analysis:
        print(f"  Pareto optimal solutions: {pareto_count}")
    print(f"  Solutions shown: {shown_count}")
    print(f"  Current generation: {generation}")


def update_pareto_optimality(solutions_by_key, all_objectives):
    """Update Pareto optimality for all solutions found so far."""
    if not all_objectives:
        return
        
    all_F = np.array(all_objectives)
    n_solutions = len(all_F)
    pareto_optimal = np.ones(n_solutions, dtype=bool)
    
    for i in range(n_solutions):
        for j in range(n_solutions):
            if i != j:
                # Check if solution j dominates solution i
                if np.all(all_F[j] <= all_F[i]) and np.any(all_F[j] < all_F[i]):
                    pareto_optimal[i] = False
                    break
    
    # Update the pareto_optimal flag in solutions_by_key
    solution_keys = list(solutions_by_key.keys())
    for i, key in enumerate(solution_keys):
        if i < len(pareto_optimal):
            solutions_by_key[key]['pareto_optimal'] = pareto_optimal[i]


def update_pareto_optimality_for_display(solutions_by_key, all_objectives):
    """Update Pareto optimality for display purposes (scatter plot and table)."""
    if not all_objectives:
        return
        
    all_F = np.array(all_objectives)
    n_solutions = len(all_F)
    pareto_optimal = np.ones(n_solutions, dtype=bool)
    
    for i in range(n_solutions):
        for j in range(n_solutions):
            if i != j:
                # Check if solution j dominates solution i
                if np.all(all_F[j] <= all_F[i]) and np.any(all_F[j] < all_F[i]):
                    pareto_optimal[i] = False
                    break
    
    # Update the pareto_optimal flag in solutions_by_key for display purposes
    solution_keys = list(solutions_by_key.keys())
    for i, key in enumerate(solution_keys):
        if i < len(pareto_optimal):
            solutions_by_key[key]['pareto_optimal'] = pareto_optimal[i]


def initialize_csv(
    csv_filename,
    metadata,
    variable_names=None,
    problem_type='envelope',
    constrained_analysis=False,
    overheating_criterion='ASHRAE'
):
    """Initialize CSV headers for envelope or shading."""
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        csvfile.write('# Optimization Metadata\n')
        for key, value in (metadata or {}).items():
            csvfile.write('# %s: %s\n' % (key, value))
        csvfile.write('#\n')

        fieldnames = _csv_fieldnames(problem_type, variable_names, constrained_analysis, overheating_criterion)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


# Helper functions for CSV header/row construction

def _objective_headers(problem_type, constrained_analysis, overheating_criterion):
    if problem_type == 'shading':
        return [
            'Mechanical_Energy_Use_Intensity',
            'Max_Living_Room_Overheating',
            'Max_Bedroom_Overheating'
        ]
    if constrained_analysis:
        if overheating_criterion.upper() == 'CIBSE59':
            return [
                'Mechanical_Energy_Use_Intensity',
                'Max_Living_Room_Overheating',
                'Max_Bedroom_Overheating'
            ]
        return ['Mechanical_Energy_Use_Intensity', 'Max_Overheating_Hours']
    return ['Mechanical_Energy_Use_Intensity', 'Average_Overheating_Hours']


def _csv_fieldnames(problem_type, variable_names, constrained_analysis, overheating_criterion):
    base = ['Generation', 'Iteration']
    objectives = _objective_headers(problem_type, constrained_analysis, overheating_criterion)
    if problem_type == 'shading':
        return base + ['Shading_Schedule'] + objectives + ['Pareto_Optimal']
    vars_part = variable_names or []
    tail = [] if constrained_analysis else ['Pareto_Optimal']
    return base + vars_part + objectives + tail


def _format_row(problem_type, result, variable_names, constrained_analysis, overheating_criterion, mode='run'):
    # mode: 'run' (during optimization) or 'final' (after Pareto marking)
    row = {
        'Generation': result.get('generation') or 0,
        'Iteration': result.get('iteration') or 0
    }

    if problem_type == 'shading':
        schedule_str = ''.join('1' if v == 1 else '0' for v in result.get('shading_schedule', []))
        row['Shading_Schedule'] = schedule_str
        row['Mechanical_Energy_Use_Intensity'] = '%.6f' % result.get('mechanical_energy_use_intensity', 0.0)
        row['Max_Living_Room_Overheating'] = '%.3f' % result.get('max_living_room_overheating', 0.0)
        row['Max_Bedroom_Overheating'] = '%.3f' % result.get('max_bedroom_overheating', 0.0)
        row['Pareto_Optimal'] = '' if mode == 'run' else ('Yes' if result.get('pareto_optimal', False) else 'No')
        return row

    # envelope
    if variable_names:
        for var_name in variable_names:
            if var_name in result:
                row[var_name] = result[var_name]

    row['Mechanical_Energy_Use_Intensity'] = '%.4f' % result.get('mechanical_energy_use_intensity', 0.0)

    if constrained_analysis:
        if overheating_criterion.upper() == 'CIBSE59':
            row['Max_Living_Room_Overheating'] = '%.2f' % result.get('max_living_room_overheating', 0.0)
            row['Max_Bedroom_Overheating'] = '%.2f' % result.get('max_bedroom_overheating', 0.0)
        else:
            row['Max_Overheating_Hours'] = '%.2f' % result.get('max_overheating_hours', 0.0)
        return row

    # unconstrained envelope
    row['Average_Overheating_Hours'] = '%.2f' % result.get('average_overheating_hours', 0.0)
    row['Pareto_Optimal'] = 'Pending' if mode == 'run' else ('Yes' if result.get('pareto_optimal', False) else 'No')
    return row


def save_to_csv(
    csv_filename, 
    solutions_by_generation, 
    solutions_by_key, 
    variable_names, 
    generation,
    problem_type='envelope',
    constrained_analysis=False,
    overheating_criterion='ASHRAE'
):
    """General CSV save function that works for both envelope and shading problems."""
    if generation not in solutions_by_generation:
        return

    fieldnames = _csv_fieldnames(problem_type, variable_names, constrained_analysis, overheating_criterion)
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for combination_key in solutions_by_generation[generation]:
            result = solutions_by_key[combination_key]
            row = _format_row(problem_type, result, variable_names, constrained_analysis, overheating_criterion, mode='run')
            writer.writerow(row)


def save_final_results(
    csv_filename, 
    solutions_by_key, 
    variable_names, 
    metadata, 
    problem_type='envelope',
    constrained_analysis=False,
    overheating_criterion='ASHRAE'
):
    """General final results save function."""
    all_solutions = list(solutions_by_key.values())
    if not all_solutions:
        return

    # Calculate Pareto optimality for unconstrained problems
    if not constrained_analysis:
        if problem_type == 'shading':
            F = np.array([
                [
                    s.get('mechanical_energy_use_intensity', 0.0),
                    s.get('max_living_room_overheating', 0.0),
                    s.get('max_bedroom_overheating', 0.0),
                ]
                for s in all_solutions
            ])
        else:
            F = np.array([
                [
                    s.get('mechanical_energy_use_intensity', 0.0),
                    s.get('average_overheating_hours', 0.0)
                ]
                for s in all_solutions
            ])

        n = len(F)
        pareto = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    pareto[i] = False
                    break
        for solution, is_pareto in zip(all_solutions, pareto):
            solution['pareto_optimal'] = is_pareto

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        csvfile.write('# Optimization Metadata\n')
        for key, value in (metadata or {}).items():
            csvfile.write('# %s: %s\n' % (key, value))
        csvfile.write('#\n')

        fieldnames = _csv_fieldnames(problem_type, variable_names, constrained_analysis, overheating_criterion)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for s in all_solutions:
            row = _format_row(problem_type, s, variable_names, constrained_analysis, overheating_criterion, mode='final')
            writer.writerow(row)

    print("\nOptimization complete! Results saved to: %s" % csv_filename)
    if not constrained_analysis:
        print("Pareto optimal solutions have been identified and marked in the CSV file.")


def initialize_plots():
    """Initialize the two plots that will be updated during optimization."""
    # Create two figures with specific names for identification
    heatmap_fig = plt.figure('Shading Strategy', figsize=(10, 12))
    evolution_fig = plt.figure('Evolution', figsize=(8, 10))
    
    # Clear any existing content
    heatmap_fig.clear()
    evolution_fig.clear()
    
    # Create axes for evolution plot
    evolution_axes = evolution_fig.subplots(3, 1)
    evolution_fig.suptitle('Evolution of Solutions', fontsize=14)
    
    # Set up evolution plot axes
    evolution_axes[0].set_ylabel('MEUI (kWh/m²)')
    evolution_axes[0].grid(True)
    
    evolution_axes[1].set_ylabel('Living Room Overheating (hrs)')
    evolution_axes[1].grid(True)
    
    evolution_axes[2].set_ylabel('Bedroom Overheating (hrs)')
    evolution_axes[2].set_xlabel('Generation')
    evolution_axes[2].grid(True)
    
    # Create axes for heatmap
    heatmap_axes = heatmap_fig.subplots(3, 2)
    heatmap_fig.suptitle('Shading Strategy (Minimizing Total Overheating Hours)', fontsize=16, y=0.95)
    
    # Show the figures
    heatmap_fig.show()
    evolution_fig.show()
    
    return heatmap_fig, heatmap_axes, evolution_fig, evolution_axes


def plot_shading_heat_map(best_solution, fig=None, axes=None):
    """
    Plot and display the best shading strategy as a heatmap for each shading element.
    Updates the display in real-time during optimization.
    
    Args:
        best_solution: List of 144 binary values representing the best shading strategy
        fig: Figure to update (if None, creates new figure)
        axes: Axes to update (if None, creates new axes)
    """
    if fig is None or axes is None:
        return
    
    # Define labels
    shading_elements = [
        'South Overhang', 'South Fin',
        'West Overhang', 'West Fin',
        'East Overhang', 'East Fin'
    ]
    seasons = ['Summer', 'Spring', 'Winter', 'Fall']
    time_periods = [
        '6-8am', '8-10am', '10am-12pm',
        '12-2pm', '2-4pm', '4-6pm'
    ]
    
    # Plot each shading element
    for idx, (ax, element) in enumerate(zip(axes.flatten(), shading_elements)):
        # Clear the axis
        ax.clear()
        
        # Extract data for this element (24 values: 4 seasons × 6 time periods)
        start_idx = idx * 24
        element_data = best_solution[start_idx:start_idx + 24]
        
        # Reshape into 4×6 matrix (seasons × time periods)
        data_matrix = np.array(element_data).reshape(4, 6)
        
        # Create heatmap
        sns.heatmap(
            data_matrix,
            ax=ax,
            cmap='RdYlBu_r',  # Red for shading deployed, white for retracted
            cbar=False,
            xticklabels=time_periods,
            yticklabels=seasons,
            linewidths=0.5,
            linecolor='gray',
            square=True,
            annot=True,  # Show values in cells
            fmt='d',     # Integer format for annotations
            cbar_kws={'label': 'Shading Deployed'}
        )
        
        # Customize subplot
        ax.set_title(element, pad=10)
        ax.set_ylabel('Season')
        
        # Only show x-axis labels and title for bottom two subplots (indices 4 and 5)
        if idx in [4, 5]:
            ax.set_xlabel('Time of Day')
            ax.set_xticklabels(time_periods, rotation=45, ha='right')
        else:
            ax.set_xticklabels([])
    
    # Update title with more space from subplots
    fig.suptitle('Shading Strategy (Minimizing Total Overheating Hours)', fontsize=16, y=0.98)
    
    # Adjust layout and update display with padding for title
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)  # Small pause to allow the plot to update


def plot_evolution(solutions_by_key, top_n=5, fig=None, axes=None):
    """
    Plot and display the evolution of shading strategies across generations.
    Shows the top N solutions by generation. Updates in real-time during optimization.
    
    Args:
        solutions_by_key: Dictionary of solutions from optimization
        top_n: Number of top solutions to show per generation
        fig: Figure to update (if None, creates new figure)
        axes: Axes to update (if None, creates new axes)
    """
    if fig is None or axes is None:
        return
    
    # Group solutions by generation
    solutions_by_gen = {}
    for solution in solutions_by_key.values():
        gen = solution['generation']
        if gen not in solutions_by_gen:
            solutions_by_gen[gen] = []
        solutions_by_gen[gen].append(solution)
    
    # Calculate metrics for each generation
    generations = sorted(solutions_by_gen.keys())
    if not generations:
        return
    
    # Clear all axes
    for ax in axes:
        ax.clear()
    
    # Plot MEUI evolution
    avg_meui = [np.mean([s['mechanical_energy_use_intensity'] for s in solutions_by_gen[gen][:top_n]]) for gen in generations]
    best_meui = [min([s['mechanical_energy_use_intensity'] for s in solutions_by_gen[gen]]) for gen in generations]
    axes[0].plot(generations, avg_meui, 'b-', label=f'Avg Top {top_n}')
    axes[0].plot(generations, best_meui, 'b--', label='Best')
    axes[0].set_ylabel('MEUI (kWh/m²)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Living Room Overheating evolution
    avg_lr = [np.mean([s['max_living_room_overheating'] for s in solutions_by_gen[gen][:top_n]]) for gen in generations]
    best_lr = [min([s['max_living_room_overheating'] for s in solutions_by_gen[gen]]) for gen in generations]
    axes[1].plot(generations, avg_lr, 'r-', label=f'Avg Top {top_n}')
    axes[1].plot(generations, best_lr, 'r--', label='Best')
    axes[1].set_ylabel('Living Room Overheating (hrs)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot Bedroom Overheating evolution
    avg_br = [np.mean([s['max_bedroom_overheating'] for s in solutions_by_gen[gen][:top_n]]) for gen in generations]
    best_br = [min([s['max_bedroom_overheating'] for s in solutions_by_gen[gen]]) for gen in generations]
    axes[2].plot(generations, avg_br, 'g-', label=f'Avg Top {top_n}')
    axes[2].plot(generations, best_br, 'g--', label='Best')
    axes[2].set_xlabel('Generation')
    axes[2].set_ylabel('Bedroom Overheating (hrs)')
    axes[2].legend()
    axes[2].grid(True)
    
    # Update title
    fig.suptitle(f'Evolution of Top {top_n} Solutions', fontsize=14)
    
    # Adjust layout and update display
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)  # Small pause to allow the plot to update


def print_final_results(res, shared_cache):
    """Print final optimization results."""
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print("Best solution found: %s" % res.X)
    print("Function values: %s" % res.F)
    print("Number of evaluations: %s" % res.algorithm.evaluator.n_eval)
    
    # Print cache statistics
    print("\nFinal shared cache statistics:")
    print(f"  Total cached combinations: {len(shared_cache)}")


def generate_dummy_shading_results(shading_schedule):
    """Generate dummy simulation results for shading optimization testing."""
    
    # Simulate some computation time
    time.sleep(0.01)  # Small delay to simulate computation
    
    # Generate dummy results based on shading schedule
    # Convert shading schedule to a numeric value for calculation
    shading_sum = sum(shading_schedule)
    shading_ratio = shading_sum / len(shading_schedule)
    
    # Generate realistic dummy results
    # More shading (higher ratio) should generally reduce overheating but may increase energy use
    base_meui = 80.0  # Base mechanical energy use intensity
    base_overheating = 25.0  # Base overheating hours
    
    # Adjust based on shading schedule
    meui_variation = random.uniform(-10, 15)  # Random variation
    overheating_variation = random.uniform(-8, 5)  # Random variation
    
    # Shading effect: more shading reduces overheating but may slightly increase energy use
    meui = base_meui + meui_variation + (shading_ratio * 20)  # More shading = slightly higher energy use
    average_overheating = max(0, base_overheating + overheating_variation - (shading_ratio * 15))  # More shading = less overheating
    
    # CIBSE59 specific values
    max_living_room_overheating = max(0, random.uniform(10, 35) - (shading_ratio * 10))
    max_bedroom_overheating = max(0, random.uniform(5, 25) - (shading_ratio * 8))
    
    return {
        'success': True,
        'district_heating_kwh': meui * 2590.29,  # Convert back to kWh
        'average_overheating_hours': average_overheating,
        'max_living_room_overheating': max_living_room_overheating,
        'max_bedroom_overheating': max_bedroom_overheating
    }


def print_generation_summary(generation, total_evaluations, new_evaluations, skipped_evaluations, cache_size):
    """Print generation summary information."""
    print(f"\nGeneration {generation} Summary:")
    print(f"  Total individuals in generation: {total_evaluations}")
    print(f"  New unique evaluations: {new_evaluations}")
    print(f"  Duplicate evaluations skipped: {skipped_evaluations}")
    print(f"  Total unique solutions in cache: {cache_size}")


def print_optimization_parameters(params_dict, algorithm_name="NSGA-II"):
    """Print optimization parameters in a standardized format."""
    print("\nOptimization Parameters:")
    for key, value in params_dict.items():
        print(f"  {key}: {value}")
    print(f"  Algorithm: {algorithm_name}")
    print("-" * 50)


def print_shading_optimization_table(solutions_by_key):
    """Print a table with all evaluated shading solutions."""
    print("\n" + "="*130)
    print("SHADING OPTIMIZATION RESULTS TABLE")
    print("="*130)
    print(f"{'Gen':<4} {'Iter':<5} {'Shading Schedule':<50} {'MEUI':<10} {'Max_LR':<8} {'Max_BR':<8} {'Pareto':<6}")
    print("-"*130)
    
    # Get all solutions as a list for processing
    all_solutions = []
    for combination_key, solution in solutions_by_key.items():
        all_solutions.append(solution)
    
    # Calculate Pareto optimality
    objectives = np.array([[s['mechanical_energy_use_intensity'], 
                          s['max_living_room_overheating'],
                          s['max_bedroom_overheating']] for s in all_solutions])
    pareto_mask = np.ones(len(objectives), dtype=bool)
    
    for i in range(len(objectives)):
        for j in range(len(objectives)):
            if i != j:
                # Check if solution j dominates solution i
                if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    pareto_mask[i] = False
                    break
    
    # Mark Pareto optimality in solutions
    for solution, is_pareto in zip(all_solutions, pareto_mask):
        solution['pareto_optimal'] = is_pareto
    
    # Get Pareto optimal solutions
    pareto_solutions = [s for s in all_solutions if s['pareto_optimal']]
    
    # Get last 100 non-Pareto solutions
    non_pareto_solutions = [s for s in all_solutions if not s['pareto_optimal']]
    last_100_non_pareto = sorted(non_pareto_solutions, 
                                key=lambda x: (x['generation'], x['iteration']))[-100:]
    
    # Print Pareto optimal solutions first
    if pareto_solutions:
        print("PARETO OPTIMAL SOLUTIONS:")
        for solution in sorted(pareto_solutions, key=lambda x: (x['generation'], x['iteration'])):
            gen = solution['generation']
            iteration = solution['iteration']
            shading_schedule = solution['shading_schedule']
            meui = solution['mechanical_energy_use_intensity']
            max_lr = solution['max_living_room_overheating']
            max_br = solution['max_bedroom_overheating']
            
            # Convert shading schedule to string representation
            schedule_str = ''.join(['1' if x == 1 else '0' for x in shading_schedule[:20]])
            if len(shading_schedule) > 20:
                schedule_str += "..."
            
            print(f"{gen:<4} {iteration:<5} {schedule_str:<50} {meui:<10.2f} {max_lr:<8.1f} {max_br:<8.1f} {'✓':<6}")
        
        print("-"*130)
    
    # Print last 100 non-Pareto solutions
    if last_100_non_pareto:
        print("LAST 100 NON-PARETO SOLUTIONS:")
        for solution in last_100_non_pareto:
            gen = solution['generation']
            iteration = solution['iteration']
            shading_schedule = solution['shading_schedule']
            meui = solution['mechanical_energy_use_intensity']
            max_lr = solution['max_living_room_overheating']
            max_br = solution['max_bedroom_overheating']
            
            # Convert shading schedule to string representation
            schedule_str = ''.join(['1' if x == 1 else '0' for x in shading_schedule[:20]])
            if len(shading_schedule) > 20:
                schedule_str += "..."
            
            print(f"{gen:<4} {iteration:<5} {schedule_str:<50} {meui:<10.2f} {max_lr:<8.1f} {max_br:<8.1f} {' ':<6}")
    
    print("-"*130)
    
    # Print summary
    print("\nSummary:")
    print(f"  Total solutions: {len(all_solutions)}")
    print(f"  Pareto optimal solutions: {len(pareto_solutions)}")
    print(f"  Non-Pareto solutions shown: {len(last_100_non_pareto)}")


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