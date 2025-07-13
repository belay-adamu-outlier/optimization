"""
CLI Utilities for Building Envelope Optimization

This module handles all command-line interface logic including:
- Interactive argument collection
- User prompts and validation
- Parameter summary and confirmation
"""

import sys


def get_interactive_arguments():
    """Get all arguments interactively with clear default value prompts."""
    print("=" * 60)
    print("BUILDING ENVELOPE OPTIMIZATION - INTERACTIVE SETUP")
    print("=" * 60)
    print("Please provide the following parameters. Press Enter to use default values.\n")
    
    # Weather file selection
    weather_file = _get_weather_file_choice()
    print(f"Selected weather file: {weather_file}\n")
    
    # Number of generations
    generations = _get_numeric_input("Enter number of generations", 100, "generations")
    print(f"Number of generations: {generations}\n")
    
    # Population size
    population_size = _get_numeric_input("Enter population size", 50, "population size")
    print(f"Population size: {population_size}\n")
    
    # Crossover probability
    crossover_prob = _get_float_input("Enter crossover probability", 0.5, "crossover probability")
    print(f"Crossover probability: {crossover_prob}\n")
    
    # Mutation probability
    mutation_prob = _get_float_input("Enter mutation probability", 0.125, "mutation probability")
    print(f"Mutation probability: {mutation_prob}\n")
    
    # Debug mode
    debug_mode = _get_boolean_input("Enable debug mode", False, "debug mode")
    print(f"Debug mode: {'Enabled' if debug_mode else 'Disabled'}\n")
    
    # Random seed
    seed = _get_seed_input()
    print(f"Random seed: {seed if seed is not None else 'Auto-generated'}\n")
    
    # Summary and confirmation
    _print_summary_and_confirm(weather_file, generations, population_size, crossover_prob, mutation_prob, debug_mode, seed)
    
    return {
        'weather': weather_file,
        'generations': generations,
        'population_size': population_size,
        'crossover_prob': crossover_prob,
        'mutation_prob': mutation_prob,
        'debug': debug_mode,
        'seed': seed
    }


def _get_weather_file_choice():
    """Get weather file choice from user."""
    print("Weather file options:")
    print("1. Vancouver - 2021 AMY.epw (default)")
    print("2. CAN_BC_Vancouver.718920_CWEC.epw")
    print("3. Enter custom path")
    
    choice = input("Enter choice (1-3) or press Enter for default value of 1: ").strip()
    
    if choice == "1" or choice == "":
        return "Vancouver - 2021 AMY.epw"
    elif choice == "2":
        return "CAN_BC_Vancouver.718920_CWEC.epw"
    elif choice == "3":
        custom_path = input("Enter weather file path: ").strip()
        if not custom_path:
            print("Invalid path, using default: Vancouver - 2021 AMY.epw")
            return "Vancouver - 2021 AMY.epw"
        return custom_path
    else:
        print("Invalid choice, using default: Vancouver - 2021 AMY.epw")
        return "Vancouver - 2021 AMY.epw"


def _get_numeric_input(prompt, default_value, param_name):
    """Get numeric input from user with validation."""
    input_text = input(f"{prompt} (press Enter for default value of {default_value}): ").strip()
    
    if not input_text:
        return default_value
    
    try:
        value = int(input_text)
        if value <= 0:
            print(f"Invalid input, using default: {default_value}")
            return default_value
        return value
    except ValueError:
        print(f"Invalid input, using default: {default_value}")
        return default_value


def _get_float_input(prompt, default_value, param_name):
    """Get float input from user with validation."""
    input_text = input(f"{prompt} (press Enter for default value of {default_value}): ").strip()
    
    if not input_text:
        return default_value
    
    try:
        value = float(input_text)
        if value < 0 or value > 1:
            print(f"Invalid input (must be between 0 and 1), using default: {default_value}")
            return default_value
        return value
    except ValueError:
        print(f"Invalid input, using default: {default_value}")
        return default_value


def _get_boolean_input(prompt, default_value, param_name):
    """Get boolean input from user with validation."""
    default_text = "y" if default_value else "n"
    input_text = input(f"{prompt}? (y/n, press Enter for default value of {default_text}): ").strip().lower()
    
    if not input_text:
        return default_value
    
    return input_text in ['y', 'yes']


def _get_seed_input():
    """Get random seed input from user with validation."""
    input_text = input("Enter random seed (press Enter for auto-generated): ").strip()
    
    if not input_text:
        return None  # Auto-generate seed
    
    try:
        seed = int(input_text)
        if seed < 0:
            print("Invalid seed (must be non-negative), using auto-generated seed")
            return None
        return seed
    except ValueError:
        print("Invalid seed (must be integer), using auto-generated seed")
        return None


def _print_summary_and_confirm(weather_file, generations, population_size, crossover_prob, mutation_prob, debug_mode, seed):
    """Print parameter summary and get user confirmation."""
    print("=" * 60)
    print("OPTIMIZATION PARAMETERS SUMMARY")
    print("=" * 60)
    print(f"Weather File: {weather_file}")
    print(f"Number of Generations: {generations}")
    print(f"Population Size: {population_size}")
    print(f"Crossover Probability: {crossover_prob}")
    print(f"Mutation Probability: {mutation_prob}")
    print(f"Debug Mode: {'Enabled' if debug_mode else 'Disabled'}")
    print(f"Random Seed: {seed if seed is not None else 'Auto-generated'}")
    print("=" * 60)
    
    # Confirm before proceeding
    confirm = input("\nProceed with these parameters? (y/n, press Enter for yes): ").strip().lower()
    if confirm in ['n', 'no']:
        print("Setup cancelled. Exiting...")
        sys.exit(0)
    
    print("\nStarting optimization...\n")


def print_optimization_header():
    """Print the optimization header."""
    print("=" * 80)
    print("BUILDING ENVELOPE OPTIMIZATION USING NSGA-II")
    print("=" * 80)


def print_optimization_footer():
    """Print the optimization footer."""
    print("=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80) 