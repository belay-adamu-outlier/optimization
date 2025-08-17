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


def get_facade_orientation(element_name):
    """
    Determine the facade orientation from an element name.
    
    Args:
        element_name (str): Name of the shading element
        
    Returns:
        str: Facade orientation ('south', 'west', 'east', 'north', or 'other')
    """
    name = element_name.lower()
    if name.startswith('south_'):
        return 'south'
    if name.startswith('west_'):
        return 'west'
    if name.startswith('east_'):
        return 'east'
    if name.startswith('north_'):
        return 'north'
    return 'other'


def get_facades_present(active_elements):
    """
    Get list of facades present in active elements, ordered by priority.
    
    Args:
        active_elements (list): List of active element names
        
    Returns:
        list: Ordered list of facades present (south, west, east, north)
    """
    facades_order = ['south', 'west', 'east', 'north']
    facades_present = [f for f in facades_order if any(get_facade_orientation(e) == f for e in active_elements)]
    return facades_present


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
    pareto_solutions = []
    if not constrained_analysis:
        pareto_solutions = [result for result in all_solutions if result.get('pareto_optimal', False)]
    
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
            pareto_mark = "✓" if result.get('pareto_optimal', False) else " "
        
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
                f"{result['thermal_energy_demand_intensity']:<8.3f}",
                f"{result.get('max_living_room_overheating', 0.0):<12.1f}",
                f"{result.get('max_bedroom_overheating', 0.0):<12.1f}"
            ])
        else:
            line_parts.extend([
                f"{result['thermal_energy_demand_intensity']:<8.3f}",
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


def _calculate_evolution_data(solutions_by_gen, generations, top_n, metric_key):
    """
    Calculate evolution data for a given metric.
    
    Args:
        solutions_by_gen: Dict mapping generation to list of solutions
        generations: List of generation numbers
        top_n: Number of best solutions to average for previous generations
        metric_key: Key to extract metric from solution dict
        
    Returns:
        tuple: (avg_values, cumulative_best_values, final_best_value)
    """
    avg_values = []
    cumulative_best_values = []
    
    # Calculate cumulative best at each generation (progressive improvement)
    running_best = float('inf')  # Start with worst possible value
    
    for gen in generations:
        # Average of ALL solutions in each generation (shows true evolution)
        gen_avg = np.mean([s[metric_key] for s in solutions_by_gen[gen]])
        avg_values.append(gen_avg)
        
        # Find best value in current generation
        gen_best = min([s[metric_key] for s in solutions_by_gen[gen]])
        
        # Update running best if current generation has better value
        running_best = min(running_best, gen_best)
        cumulative_best_values.append(running_best)
    
    return avg_values, cumulative_best_values, running_best


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
    overheating_criterion='ASHRAE',
    objective_names=None,
    multiobjective=True
):
    """Initialize CSV headers for envelope or shading.

    objective_names: list of objective column names (for shading).
    multiobjective: include Pareto column only if True.
    """
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        csvfile.write('# Optimization Metadata\n')
        for key, value in (metadata or {}).items():
            csvfile.write('# %s: %s\n' % (key, value))
        csvfile.write('#\n')

        fieldnames = _csv_fieldnames(
            problem_type,
            variable_names,
            constrained_analysis,
            overheating_criterion,
            objective_names=objective_names,
            multiobjective=multiobjective
        )
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


# Helper functions for CSV header/row construction

def _objective_headers(problem_type, constrained_analysis, overheating_criterion):
    if problem_type == 'shading':
        return [
            'Thermal_Energy_Demand_Intensity',
            'Max_Living_Room_Overheating',
            'Max_Bedroom_Overheating'
        ]
    if constrained_analysis:
        if overheating_criterion.upper() == 'CIBSE59':
            return [
                'Thermal_Energy_Demand_Intensity',
                'Max_Living_Room_Overheating',
                'Max_Bedroom_Overheating'
            ]
        return ['Thermal_Energy_Demand_Intensity', 'Max_Overheating_Hours']
    return ['Thermal_Energy_Demand_Intensity', 'Average_Overheating_Hours']


def _csv_fieldnames(
    problem_type,
    variable_names,
    constrained_analysis,
    overheating_criterion,
    objective_names=None,
    multiobjective=True
):
    base = ['Generation', 'Iteration']
    objectives = _objective_headers(problem_type, constrained_analysis, overheating_criterion)
    if problem_type == 'shading':
        # For shading problems, always include all three columns (TEDI + both overheating)
        # The objective_names parameter only controls which objectives are optimized, not which columns are displayed
        objs = [
            'Thermal_Energy_Demand_Intensity',
            'Max_Living_Room_Overheating',
            'Max_Bedroom_Overheating'
        ]
        tail = ['Pareto_Optimal'] if multiobjective else []
        return base + ['Shading_Schedule'] + objs + tail
    vars_part = variable_names or []
    tail = [] if constrained_analysis else ['Pareto_Optimal']
    return base + vars_part + objectives + tail


def _format_row(
    problem_type,
    result,
    variable_names,
    constrained_analysis,
    overheating_criterion,
    mode='run',
    objective_names=None,
    multiobjective=True
):
    # mode: 'run' (during optimization) or 'final' (after Pareto marking)
    row = {
        'Generation': result.get('generation') or 0,
        'Iteration': result.get('iteration') or 0
    }

    if problem_type == 'shading':
        schedule_str = ''.join('1' if v == 1 else '0' for v in result.get('shading_schedule', []))
        row['Shading_Schedule'] = schedule_str
        # For shading problems, always write all three columns regardless of objective_names
        # objective_names only controls which objectives are optimized, not which columns are displayed
        names = [
            'Thermal_Energy_Demand_Intensity',
            'Max_Living_Room_Overheating',
            'Max_Bedroom_Overheating'
        ]
        for name in names:
            val = result.get(name.lower(), None)
            if val is None:
                # try canonical keys used in code
                val = result.get('thermal_energy_demand_intensity' if 'Mechanical' in name else (
                    'max_living_room_overheating' if 'Living' in name else 'max_bedroom_overheating'
                ), 0.0)
            # format
            if 'Mechanical' in name:
                row[name] = '%.6f' % float(val)
            else:
                row[name] = '%.3f' % float(val)
        if multiobjective:
            row['Pareto_Optimal'] = '' if mode == 'run' else (
                'Yes' if result.get('pareto_optimal', False) else 'No')
        return row

    # envelope
    if variable_names:
        for var_name in variable_names:
            if var_name in result:
                row[var_name] = result[var_name]

    row['Thermal_Energy_Demand_Intensity'] = '%.4f' % result.get('thermal_energy_demand_intensity', 0.0)

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
    overheating_criterion='ASHRAE',
    objective_names=None,
    multiobjective=True
):
    """General CSV save function that works for both envelope and shading problems."""
    if generation not in solutions_by_generation:
        return

    fieldnames = _csv_fieldnames(
        problem_type,
        variable_names,
        constrained_analysis,
        overheating_criterion,
        objective_names=objective_names,
        multiobjective=multiobjective
    )
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for combination_key in solutions_by_generation[generation]:
            result = solutions_by_key[combination_key]
            row = _format_row(
                problem_type,
                result,
                variable_names,
                constrained_analysis,
                overheating_criterion,
                mode='run',
                objective_names=objective_names,
                multiobjective=multiobjective
            )
            writer.writerow(row)


def save_final_results(
    csv_filename, 
    solutions_by_key, 
    variable_names, 
    metadata, 
    problem_type='envelope',
    constrained_analysis=False,
    overheating_criterion='ASHRAE',
    objective_names=None,
    multiobjective=True
):
    """General final results save function."""
    all_solutions = list(solutions_by_key.values())
    if not all_solutions:
        return

    # Calculate Pareto optimality for unconstrained problems
    if not constrained_analysis and multiobjective:
        if problem_type == 'shading':
            # Use the selected objective names to build F in order
            names = objective_names or [
                'Thermal_Energy_Demand_Intensity',
                'Max_Living_Room_Overheating',
                'Max_Bedroom_Overheating'
            ]
            key_map = {
                'Thermal_Energy_Demand_Intensity': 'thermal_energy_demand_intensity',
                'Max_Living_Room_Overheating': 'max_living_room_overheating',
                'Max_Bedroom_Overheating': 'max_bedroom_overheating'
            }
            F = np.array([
                [s.get(key_map.get(name, name), 0.0) for name in names]
                for s in all_solutions
            ])
        else:
            F = np.array([
                [
                    s.get('thermal_energy_demand_intensity', 0.0),
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

        fieldnames = _csv_fieldnames(
            problem_type,
            variable_names,
            constrained_analysis,
            overheating_criterion,
            objective_names=objective_names,
            multiobjective=multiobjective
        )
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for s in all_solutions:
            row = _format_row(
                problem_type,
                s,
                variable_names,
                constrained_analysis,
                overheating_criterion,
                mode='final',
                objective_names=objective_names,
                multiobjective=multiobjective
            )
            writer.writerow(row)

    print("\nOptimization complete! Results saved to: %s" % csv_filename)
    if not constrained_analysis:
        print("Pareto optimal solutions have been identified and marked in the CSV file.")


def initialize_plots():
    """Initialize plots used during optimization.

    Only the evolution plot is created here. The heatmap figure is managed
    elsewhere to avoid duplicate/overlapping axes and spacing issues.
    """
    # Evolution figure
    evolution_fig = plt.figure('Evolution', figsize=(8, 10))
    evolution_fig.clear()
    evolution_axes = evolution_fig.subplots(3, 1)
    evolution_fig.suptitle('Evolution of Solutions', fontsize=14)
    
    # Set up evolution plot axes
    evolution_axes[0].set_ylabel('TEDI (kWh/m²)')
    evolution_axes[0].grid(True)
    
    evolution_axes[1].set_ylabel('Living Room Overheating (hrs)')
    evolution_axes[1].grid(True)
    
    evolution_axes[2].set_ylabel('Bedroom Overheating (hrs)')
    evolution_axes[2].set_xlabel('Generation')
    evolution_axes[2].grid(True)
    
    evolution_fig.show()
    
    # Return placeholders for heatmap (managed by the callback)
    return None, None, evolution_fig, evolution_axes


def plot_shading_heat_map(schedule_flat, active_elements=None, fig=None, axes=None):
    """
    Plot and display shading strategy as a heatmap for each shading element.
    Updates the display in real-time during optimization.
    
    Args:
        schedule_flat: List of binary values representing the shading strategy
        active_elements: List of element names in order (if None, uses default 6 elements)
        fig: Figure to update (if None, creates new figure)
        axes: Axes to update (if None, creates new axes)
    """
    if fig is None or axes is None:
        return
    
    # Use default elements if none provided (for backward compatibility)
    if active_elements is None:
        active_elements = [
            'south_overhang', 'south_fin',
            'west_overhang', 'west_fin',
            'east_overhang', 'east_fin'
        ]
    
    # Define labels
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    time_periods = [
        '6-8am', '8-10am', '10am-12pm',
        '12-2pm', '2-4pm', '4-6pm'
    ]
    
    # Group elements by facade and pair overhang/fin side-by-side per row
    facade_to_elements = {}
    for i, name in enumerate(active_elements):
        facade_to_elements.setdefault(get_facade_orientation(name), []).append((i, name))
    
    ordered_facades = get_facades_present(active_elements)
    row = 0
    
    for f in ordered_facades:
        elems = facade_to_elements[f]
        # ensure overhang first, then fin if both exist
        elems_sorted = sorted(elems, key=lambda t: (0 if 'overhang' in t[1] else 1, t[1]))
        
        for col in range(2):
            ax_idx = row*2 + col
            ax = axes[ax_idx] if ax_idx < len(axes) else None
            if ax is None:
                continue
                
            ax.clear()
            
            # Hide spines for a cleaner heatmap look
            try:
                for spine in ax.spines.values():
                    spine.set_visible(False)
            except Exception:
                pass
                
            if col < len(elems_sorted):
                elem_idx, elem_name = elems_sorted[col]
                start = elem_idx * 24
                block = (schedule_flat[start:start + 24] or [])
                if len(block) < 24:
                    block = (block + [0] * 24)[:24]
                    
                matrix = [block[0:6], block[6:12], block[12:18], block[18:24]]
                data = np.array(matrix)
                
                # Create heatmap
                sns.heatmap(
                    data,
                    ax=ax,
                    cmap='RdYlBu_r',  # Red for shading deployed, white for retracted
                    cbar=False,
                    xticklabels=time_periods if row == len(ordered_facades)-1 else ['']*6,
                    yticklabels=seasons,
                    linewidths=0.5,
                    linecolor='gray',
                    square=True,
                    annot=True,  # Show values in cells
                    fmt='d',     # Integer format for annotations
                )
                
                # Remove auto tick marks and numeric scales
                from matplotlib.ticker import FixedLocator
                ax.tick_params(axis='both', which='both', length=0, labelsize=9)
                
                # Center labels in cells; suppress numeric ticks
                ax.yaxis.set_major_locator(FixedLocator(np.arange(4) + 0.5))
                if row == len(ordered_facades)-1:
                    ax.xaxis.set_major_locator(FixedLocator(np.arange(6) + 0.5))
                else:
                    ax.xaxis.set_major_locator(FixedLocator([]))
                
                # Place titles above axes with sufficient padding
                ax.set_title(
                    elem_name.replace('_',' ').title(),
                    pad=8,
                    fontsize=11
                )
                ax.set_xlabel('Time of Day' if row == len(ordered_facades)-1 else '')
                ax.set_ylabel('Season')
            else:
                ax.set_visible(False)
        row += 1
    
    # Hide any extra axes beyond used rows*2
    used_axes = len(ordered_facades) * 2
    for j in range(used_axes, len(axes)):
        try:
            axes[j].set_visible(False)
        except Exception:
            pass
    
    # Update title with more space from subplots
    fig.suptitle('Shading Strategy (Minimizing Total Overheating Hours)', fontsize=16, y=0.97)
    
    # Add generous vertical spacing between rows for titles
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.98,
                        hspace=0.28, wspace=0.12)
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.05)  # Small pause to allow the plot to update




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
    
    # Plot TEDI evolution using refactored function
    avg_tedi, cumulative_best_tedi, _ = _calculate_evolution_data(
        solutions_by_gen, generations, top_n, 'thermal_energy_demand_intensity'
    )
    
    # Plot evolution across all generations (shows true convergence)
    axes[0].plot(generations, avg_tedi, color='b', linestyle='-', marker='o',
                 label='All Gen: All Solutions Avg')
    
    # Cumulative best shows progressive improvement over generations
    axes[0].plot(generations, cumulative_best_tedi, color='b', linestyle='--', marker='o',
                 label='Cumulative Best (Progressive)')
    axes[0].set_ylabel('TEDI (kWh/m²)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Living Room Overheating evolution using refactored function
    avg_lr, cumulative_best_lr, _ = _calculate_evolution_data(
        solutions_by_gen, generations, top_n, 'max_living_room_overheating'
    )
    
    # Plot evolution across all generations (shows true convergence)
    axes[1].plot(generations, avg_lr, color='r', linestyle='-', marker='o',
                 label='All Gen: All Solutions Avg')
    
    # Cumulative best shows progressive improvement over generations
    axes[1].plot(generations, cumulative_best_lr, color='r', linestyle='--', marker='o',
                 label='Cumulative Best (Progressive)')
    axes[1].set_ylabel('Living Room Overheating (hrs)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot Bedroom Overheating evolution using refactored function
    avg_br, cumulative_best_br, _ = _calculate_evolution_data(
        solutions_by_gen, generations, top_n, 'max_bedroom_overheating'
    )
    
    # Plot evolution across all generations (shows true convergence)
    axes[2].plot(generations, avg_br, color='g', linestyle='-', marker='o',
                 label='All Gen: All Solutions Avg')
    
    # Cumulative best shows progressive improvement over generations
    axes[2].plot(generations, cumulative_best_br, color='g', linestyle='--', marker='o',
                 label='Cumulative Best (Progressive)')
    axes[2].set_xlabel('Generation')
    axes[2].set_ylabel('Bedroom Overheating (hrs)')
    axes[2].legend()
    axes[2].grid(True)
    
    # Update title
    fig.suptitle('Evolution: All Generations (All Solutions Avg) + Cumulative Best (Progressive)', fontsize=14)
    
    # Update display (layout already handled by subplots_adjust)
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
    base_tedi = 80.0  # Base mechanical energy use intensity
    base_overheating = 25.0  # Base overheating hours
    
    # Adjust based on shading schedule
    tedi_variation = random.uniform(-10, 15)  # Random variation
    overheating_variation = random.uniform(-8, 5)  # Random variation
    
    # Shading effect: more shading reduces overheating but may slightly increase energy use
    tedi = base_tedi + tedi_variation + (shading_ratio * 20)  # More shading = slightly higher energy use
    average_overheating = max(0, base_overheating + overheating_variation - (shading_ratio * 15))  # More shading = less overheating
    
    # CIBSE59 specific values
    max_living_room_overheating = max(0, random.uniform(10, 35) - (shading_ratio * 10))
    max_bedroom_overheating = max(0, random.uniform(5, 25) - (shading_ratio * 8))
    
    return {
        'success': True,
        'district_heating_kwh': tedi * 2590.29,  # Convert back to kWh
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


def print_shading_optimization_table(solutions_by_key, multiobjective=True, objective_names=None):
    """Print a table with all evaluated shading solutions.

    multiobjective: if True, include Pareto column and compute it.
    objective_names: control objective ordering in table output.
    """
    print("\n" + "="*130)
    print("SHADING OPTIMIZATION RESULTS TABLE")
    print("="*130)
    
    # Define consistent column widths
    col_widths = {
        'Gen': 4,
        'Iter': 5,
        'Shading Schedule': 50,
        'TEDI': 10,
        'Max_LR': 8,
        'Max_BR': 8,
        'Pareto': 6
    }
    
    # Build header with consistent widths
    header = f"{'Gen':<{col_widths['Gen']}} {('Iter'):<{col_widths['Iter']}} {'Shading Schedule':<{col_widths['Shading Schedule']}} "
    names = objective_names or ['TEDI', 'Max_LR', 'Max_BR']
    for nm in names:
        header += f"{nm:<{col_widths.get(nm, 10)}} "
    if multiobjective:
        header += f"{'Pareto':<{col_widths['Pareto']}}"
    print(header)
    print("-"*130)
    
    # Get all solutions as a list for processing
    all_solutions = []
    for combination_key, solution in solutions_by_key.items():
        all_solutions.append(solution)
    
    # Calculate Pareto optimality if requested
    if multiobjective:
        key_map = {
            'TEDI': 'thermal_energy_demand_intensity',
            'Max_LR': 'max_living_room_overheating',
            'Max_BR': 'max_bedroom_overheating'
        }
        keys = [key_map.get(n, n) for n in names]
        objectives = np.array([[s.get(k, 0.0) for k in keys] for s in all_solutions])
        pareto_mask = np.ones(len(objectives), dtype=bool)
        for i in range(len(objectives)):
            for j in range(len(objectives)):
                if i != j:
                    if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                        pareto_mask[i] = False
                        break
        for solution, is_pareto in zip(all_solutions, pareto_mask):
            solution['pareto_optimal'] = is_pareto
    
    # Get Pareto optimal solutions
    if multiobjective:
        pareto_solutions = [s for s in all_solutions if s['pareto_optimal']]
        # Get last 100 non-Pareto solutions
        non_pareto_solutions = [s for s in all_solutions if not s['pareto_optimal']]
    else:
        # For single objective, all solutions are effectively "optimal"
        pareto_solutions = []
        non_pareto_solutions = all_solutions
    last_100_non_pareto = sorted(non_pareto_solutions, 
                                key=lambda x: (x['generation'], x['iteration']))[-100:]
    
    # Print Pareto optimal solutions first
    if multiobjective and pareto_solutions:
        print("PARETO OPTIMAL SOLUTIONS:")
        for solution in sorted(pareto_solutions, key=lambda x: (x['generation'], x['iteration'])):
            gen = solution['generation']
            iteration = solution['iteration']
            shading_schedule = solution['shading_schedule']
            tedi = solution['thermal_energy_demand_intensity']
            max_lr = solution['max_living_room_overheating']
            max_br = solution['max_bedroom_overheating']
            
            # Convert shading schedule to string representation
            schedule_str = ''.join(['1' if x == 1 else '0' for x in shading_schedule[:20]])
            if len(shading_schedule) > 20:
                schedule_str += "..."
            
            # Use ASCII to avoid Windows console encoding issues
            row = f"{gen:<{col_widths['Gen']}} {iteration:<{col_widths['Iter']}} {schedule_str:<{col_widths['Shading Schedule']}} {tedi:<{col_widths['TEDI']}.2f} {max_lr:<{col_widths['Max_LR']}.1f} {max_br:<{col_widths['Max_BR']}.1f}"
            row += f" {'Yes':<{col_widths['Pareto']}}"
            print(row)
        
        print("-"*130)
    
    # Print last 100 non-Pareto solutions
    if last_100_non_pareto:
        print("LAST 100 NON-PARETO SOLUTIONS:")
        for solution in last_100_non_pareto:
            gen = solution['generation']
            iteration = solution['iteration']
            shading_schedule = solution['shading_schedule']
            tedi = solution['thermal_energy_demand_intensity']
            max_lr = solution['max_living_room_overheating']
            max_br = solution['max_bedroom_overheating']
            
            # Convert shading schedule to string representation
            schedule_str = ''.join(['1' if x == 1 else '0' for x in shading_schedule[:20]])
            if len(shading_schedule) > 20:
                schedule_str += "..."
            
            row = f"{gen:<{col_widths['Gen']}} {iteration:<{col_widths['Iter']}} {schedule_str:<{col_widths['Shading Schedule']}} {tedi:<{col_widths['TEDI']}.2f} {max_lr:<{col_widths['Max_LR']}.1f} {max_br:<{col_widths['Max_BR']}.1f}"
            if multiobjective:
                row += f" {' ':<{col_widths['Pareto']}}"
            print(row)
    
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


def initialize_shading_heatmap(active_elements, figsize=(7.5, 3.5)):
    """
    Initialize a heatmap figure and axes for shading optimization visualization.
    
    Args:
        active_elements (list): List of active element names
        figsize (tuple): Base figure size, will be adjusted based on number of facades
        
    Returns:
        tuple: (figure, axes_list, facades_present)
    """
    from util.data_utils import get_facades_present
    
    # Get facades present and determine layout
    facades_present = get_facades_present(active_elements)
    rows = max(1, len(facades_present))
    cols = 2
    
    # Create figure with appropriate size
    fig = plt.figure('Shading Strategy', figsize=(figsize[0], figsize[1] * rows))
    
    # Create subplots
    ax_obj = fig.subplots(
        rows,
        cols,
        gridspec_kw={'hspace': 0.05, 'wspace': 0.12}
    )
    
    # Fine tune spacing to remove large gap between rows while
    # preserving room for the bottom x labels and left y labels
    fig.subplots_adjust(
        top=0.90, bottom=0.10, left=0.08, right=0.98,
        hspace=0.28, wspace=0.12
    )
    
    # Normalize axes to a flat list
    if isinstance(ax_obj, np.ndarray):
        axes = list(ax_obj.ravel())
    else:
        axes = [ax_obj]
    
    return fig, axes, facades_present


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])