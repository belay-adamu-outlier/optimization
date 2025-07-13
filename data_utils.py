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
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


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


def print_optimization_table(solutions_by_key, variable_names, generation):
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
    print(f"OPTIMIZATION RESULTS - PARETO OPTIMAL + LAST 100 ITERATIONS")
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
    
    header_parts.extend([
        f"{'Mech_EUI':<8}",
        f"{'Overheating':<8}",
        f"{'Pareto_Opt':<10}"
    ])
    
    header = " ".join(header_parts)
    print(header)
    print("-" * len(header))
    
    # Get all solutions as a list for processing
    all_solutions = list(solutions_by_key.values())
    
    # Get Pareto optimal solutions
    pareto_solutions = [result for result in all_solutions if result['pareto_optimal']]
    
    # Get last 100 iterations (or all if less than 100)
    last_100 = all_solutions[-100:] if len(all_solutions) > 100 else all_solutions
    
    # Combine Pareto optimal and last 100, removing duplicates
    solutions_to_show = []
    seen_combinations = set()
    
    # Add Pareto optimal solutions first
    for result in pareto_solutions:
        # Use a combination of generation and objective values as unique identifier
        combination_key = (result['generation'], result['mechanical_energy_use_intensity'], result['average_overheating_hours'])
        if combination_key not in seen_combinations:
            solutions_to_show.append(result)
            seen_combinations.add(combination_key)
    
    # Add last 100 iterations (if not already included)
    for result in last_100:
        # Use a combination of generation and objective values as unique identifier
        combination_key = (result['generation'], result['mechanical_energy_use_intensity'], result['average_overheating_hours'])
        if combination_key not in seen_combinations:
            solutions_to_show.append(result)
            seen_combinations.add(combination_key)
    
    # Sort by generation and iteration for better readability
    solutions_to_show.sort(key=lambda x: (x['generation'], x['iteration']))
    
    # Print filtered results
    for result in solutions_to_show:
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
        
        line_parts.extend([
            f"{result['mechanical_energy_use_intensity']:<8.3f}",
            f"{result['average_overheating_hours']:<8.1f}",
            f"{pareto_mark:<10}"
        ])
        
        line = " ".join(line_parts)
        print(line)
    
    # Print summary
    total_solutions = len(solutions_by_key)
    pareto_count = len(pareto_solutions)
    shown_count = len(solutions_to_show)
    print(f"\nSummary:")
    print(f"  Total solutions evaluated: {total_solutions}")
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


def initialize_csv(csv_filename, variable_names, metadata):
    """Initialize the CSV file with headers and metadata."""
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        # Write metadata section
        csvfile.write("# Optimization Metadata\n")
        for key, value in metadata.items():
            csvfile.write(f"# {key}: {value}\n")
        csvfile.write("#\n")
        
        fieldnames = ['Generation', 'Iteration'] + variable_names + ['Mechanical_Energy_Use_Intensity', 'Average_Overheating_Hours', 'Pareto_Optimal']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def save_to_csv(csv_filename, solutions_by_generation, solutions_by_key, variable_names, generation):
    """Save current generation results to CSV file."""
    if generation not in solutions_by_generation:
        return
        
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Generation', 'Iteration'] + variable_names + ['Mechanical_Energy_Use_Intensity', 'Average_Overheating_Hours', 'Pareto_Optimal']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write all results from this generation
        for combination_key in solutions_by_generation[generation]:
            result = solutions_by_key[combination_key]
            row = {
                'Generation': result.get('generation') or 0,
                'Iteration': result.get('iteration') or 0,
                'Mechanical_Energy_Use_Intensity': f"{result['mechanical_energy_use_intensity']:.4f}",
                'Average_Overheating_Hours': f"{result['average_overheating_hours']:.2f}",
                'Pareto_Optimal': 'Pending'  # Will be updated at the end
            }
            
            # Add all variable values
            for var_name in variable_names:
                if var_name in result:
                    row[var_name] = result[var_name]
            
            writer.writerow(row)


def save_final_results(csv_filename, solutions_by_key, variable_names, metadata, ax, fig):
    """Save final results with updated Pareto optimality."""
    # Update Pareto optimality one final time for final CSV
    all_objectives = []
    for result in solutions_by_key.values():
        all_objectives.append([result['mechanical_energy_use_intensity'], result['average_overheating_hours']])
    
    update_pareto_optimality(solutions_by_key, all_objectives)
    
    # Write all results to CSV with proper Pareto optimality
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        # Write metadata section
        csvfile.write("# Optimization Metadata\n")
        for key, value in metadata.items():
            csvfile.write(f"# {key}: {value}\n")
        csvfile.write("#\n")
        
        fieldnames = ['Generation', 'Iteration'] + variable_names + ['Mechanical_Energy_Use_Intensity', 'Average_Overheating_Hours', 'Pareto_Optimal']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in solutions_by_key.values():
            row = {
                'Generation': result.get('generation') or 0,
                'Iteration': result.get('iteration') or 0,
                'Mechanical_Energy_Use_Intensity': f"{result['mechanical_energy_use_intensity']:.4f}",
                'Average_Overheating_Hours': f"{result['average_overheating_hours']:.2f}",
                'Pareto_Optimal': 'Yes' if result['pareto_optimal'] else 'No'
            }
            
            # Add all variable values
            for var_name in variable_names:
                if var_name in result:
                    row[var_name] = result[var_name]
            
            writer.writerow(row)
    
    # Update title to show final results
    if ax is not None and fig is not None:
        pareto_count = sum(1 for result in solutions_by_key.values() if result['pareto_optimal'])
        ax.set_title(f'FINAL RESULTS - {pareto_count} Pareto Optimal Solutions out of {len(solutions_by_key)} Total')
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    print(f"\nOptimization complete! Results saved to: {csv_filename}")
    print(f"Pareto optimal solutions have been identified and marked in the CSV file.")


def initialize_plot():
    """Initialize the live scatter plot."""
    try:
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlabel('Mechanical Energy Use Intensity (kWh/m²)')
        ax.set_ylabel('Average Overheating Hours')
        ax.set_title('Optimization Progress - Pareto Front')
        ax.grid(True, alpha=0.3)

        # Set default axis limits so axes are visible even with no data
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Create persistent scatter plot objects
        scatter_all = ax.scatter([], [], c='blue', alpha=0.6, s=50, label='Non-Pareto Optimal')
        scatter_pareto = ax.scatter([], [], c='red', s=100, label='Pareto Optimal')

        ax.legend()
        plt.tight_layout()
        plt.show()
        print("Plot initialized successfully!")
        return fig, ax, scatter_all, scatter_pareto
    except Exception as e:
        print(f"Error initializing plot: {e}")
        return None, None, None, None


def update_plot(solutions_by_key, generation, fig, ax, scatter_all, scatter_pareto):
    """Update the scatter plot with current data."""
    if not solutions_by_key or fig is None or ax is None:
        return
    
    try:
        # Extract data from solutions_by_key
        all_solutions = list(solutions_by_key.values())
        meui_values = [result['mechanical_energy_use_intensity'] for result in all_solutions]
        overheating_values = [result['average_overheating_hours'] for result in all_solutions]
        pareto_mask = [result['pareto_optimal'] for result in all_solutions]
        
        # Separate Pareto optimal and non-Pareto optimal solutions
        pareto_meui = [meui_values[i] for i in range(len(meui_values)) if pareto_mask[i]]
        pareto_overheating = [overheating_values[i] for i in range(len(overheating_values)) if pareto_mask[i]]
        
        non_pareto_meui = [meui_values[i] for i in range(len(meui_values)) if not pareto_mask[i]]
        non_pareto_overheating = [overheating_values[i] for i in range(len(overheating_values)) if not pareto_mask[i]]
        
        # Update scatter plot data (don't recreate objects)
        if non_pareto_meui:
            scatter_all.set_offsets(np.column_stack([non_pareto_meui, non_pareto_overheating]))
        else:
            scatter_all.set_offsets(np.empty((0, 2)))
        
        if pareto_meui:
            scatter_pareto.set_offsets(np.column_stack([pareto_meui, pareto_overheating]))
        else:
            scatter_pareto.set_offsets(np.empty((0, 2)))
        
        # Auto-adjust axis limits
        if meui_values:
            x_min, x_max = min(meui_values), max(meui_values)
            y_min, y_max = min(overheating_values), max(overheating_values)
            
            # Add some padding
            x_range = x_max - x_min
            y_range = y_max - y_min
            if x_range > 0:
                ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        # Update title with generation info and Pareto count
        pareto_count = sum(pareto_mask)
        ax.set_title(f'Generation {generation} - {pareto_count} Pareto Optimal out of {len(solutions_by_key)} Total Solutions')
        
        # Update the plot with proper event loop breathing room
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)  # Give GUI framework time to handle events
        
    except Exception as e:
        print(f"Error updating plot: {e}")


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