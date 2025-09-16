"""
CLI Utilities for Building Envelope Optimization

This module handles all command-line interface logic including:
- Interactive argument collection
- User prompts and validation
- Parameter summary and confirmation
"""

import sys
import os


def get_interactive_arguments():
    """Get all arguments interactively with clear default value prompts."""
    print("=" * 60)
    print("BUILDING OPTIMIZATION - INTERACTIVE SETUP")
    print("=" * 60)
    print("Please provide the following parameters. Press Enter to use default values.\n")
    
    # Check if we're in test mode (command line arguments)
    if len(sys.argv) > 1 and '--test-mode' in sys.argv:
        return _get_test_mode_arguments()
    
    # Optimization type selection
    optimization_type = _get_optimization_type_choice()
    print(f"Optimization type: {optimization_type}\n")
    
    # Test mode selection
    test_mode = _get_boolean_input("Enable test mode (use dummy data instead of EnergyPlus simulations)", False, "test mode")
    print(f"Test mode: {'Enabled' if test_mode else 'Disabled'}\n")
    
    # Weather file selection
    weather_file = _get_weather_file_choice()
    print(f"Selected weather file: {weather_file}\n")
    
    # Constrained analysis
    constrained_analysis = _get_boolean_input("Enable constrained analysis", False, "constrained analysis")
    print(f"Constrained analysis: {'Enabled' if constrained_analysis else 'Disabled'}\n")
    
    # Overheating criterion (only if constrained analysis is enabled)
    overheating_criterion = 'ASHRAE'  # Default
    if constrained_analysis:
        overheating_criterion = _get_overheating_criterion_choice()
        print(f"Overheating criterion: {overheating_criterion}\n")
    
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
    print(f"Random seed: {seed if seed is None else 'Auto-generated'}\n")
    
    # Summary and confirmation
    _print_summary_and_confirm(
        optimization_type, 
        weather_file, 
        constrained_analysis, 
        overheating_criterion, 
        generations, 
        population_size, 
        crossover_prob, 
        mutation_prob, 
        debug_mode, 
        seed, test_mode
    )
    
    return {
        'optimization_type': optimization_type,
        'weather': weather_file,
        'constrained_analysis': constrained_analysis,
        'overheating_criterion': overheating_criterion,
        'generations': generations,
        'population_size': population_size,
        'crossover_prob': crossover_prob,
        'mutation_prob': mutation_prob,
        'debug': debug_mode,
        'seed': seed,
        'test_mode': test_mode
    }


def _get_test_mode_arguments():
    """Get arguments automatically for test mode."""
    print("TEST MODE: Using automatic test configuration\n")
    
    # Parse command line arguments for test mode
    args = sys.argv[1:]
    
    # Default test values
    test_args = {
        'optimization_type': 'shading',
        'weather': 'Vancouver - 2021 AMY.epw',
        'constrained_analysis': True,
        'overheating_criterion': 'CIBSE59',
        'generations': 2,
        'population_size': 5,
        'crossover_prob': 0.5,
        'mutation_prob': 0.125,
        'debug': False,
        'seed': None,
        'test_mode': True
    }
    
    # Override with command line arguments if provided
    for i, arg in enumerate(args):
        if arg == '--optimization-type' and i + 1 < len(args):
            test_args['optimization_type'] = args[i + 1]
        elif arg == '--generations' and i + 1 < len(args):
            test_args['generations'] = int(args[i + 1])
        elif arg == '--population-size' and i + 1 < len(args):
            test_args['population_size'] = int(args[i + 1])
        elif arg == '--constrained-analysis':
            test_args['constrained_analysis'] = True
        elif arg == '--overheating-criterion' and i + 1 < len(args):
            test_args['overheating_criterion'] = args[i + 1]
    
    print("Test configuration:")
    for key, value in test_args.items():
        print(f"  {key}: {value}")
    print()
    
    return test_args


def _get_optimization_type_choice():
    """Get optimization type choice from user."""
    print("Optimization type options:")
    print("1. Envelope optimization (default)")
    print("2. Shading optimization")
    print("3. Shading + WWR optimization")
    
    choice = input("Enter choice (1-3) or press Enter for default value of 1: ").strip()
    
    if choice == "1" or choice == "":
        return "envelope"
    elif choice == "2":
        return "shading"
    elif choice == "3":
        return "shading_wwr"
    else:
        print("Invalid choice, using default: envelope")
        return "envelope"


def _get_weather_file_choice():
    """Get weather file choice from the BEMs folder."""
    # Resolve BEMs directory relative to this file's parent directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    bems_dir = os.path.join(project_root, "BEMs")

    # Find .epw files (non-recursive)
    epw_files = []
    try:
        if os.path.isdir(bems_dir):
            for name in os.listdir(bems_dir):
                if name.lower().endswith(".epw"):
                    epw_files.append(name)
    except Exception:
        epw_files = []

    print("Weather file options (from BEMs):")
    if not epw_files:
        print("No .epw files found in BEMs.")
        print("1. Enter custom path")
        input("Enter choice (1) or press Enter to enter custom path: ")
        custom_path = input("Enter weather file path: ").strip()
        if not custom_path:
            print("No weather file provided. Please add a .epw to BEMs or enter a valid path.")
            return None
        return custom_path

    epw_files.sort()
    
    for idx, fname in enumerate(epw_files, start=1):
        label = " (default)" if idx == 1 else ""
        print(f"{idx}. {fname}{label}")
    custom_idx = len(epw_files) + 1
    
    print(f"{custom_idx}. Enter custom path")

    prompt = f"Enter choice (1-{custom_idx}) or press Enter for default value of 1: "
    choice = input(prompt).strip()

    if choice == "" or choice == "1":
        return os.path.join("BEMs", epw_files[0])

    if choice == str(custom_idx):
        custom_path = input("Enter weather file path: ").strip()
        if not custom_path:
            print(f"Invalid path, using default: {epw_files[0]}")
            return os.path.join("BEMs", epw_files[0])
        return custom_path

    try:
        idx = int(choice)
        if 1 <= idx <= len(epw_files):
            return os.path.join("BEMs", epw_files[idx - 1])
        else:
            raise ValueError()
    except (ValueError, TypeError):
        print(f"Invalid choice, using default: {epw_files[0]}")
        return os.path.join("BEMs", epw_files[0])

def _get_overheating_criterion_choice():
    """Get overheating criterion choice from user."""
    print("Overheating criterion options:")
    print("1. ASHRAE 55 Adaptive 80% (default)")
    print("2. CIBSE59")
    
    choice = input("Enter choice (1-2) or press Enter for default value of 1: ").strip()
    
    if choice == "1" or choice == "":
        return "ASHRAE"
    elif choice == "2":
        return "CIBSE59"
    else:
        print("Invalid choice, using default: ASHRAE")
        return "ASHRAE"


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


def _print_summary_and_confirm(
    optimization_type, 
    weather_file, 
    constrained_analysis, 
    overheating_criterion, 
    generations, 
    population_size, 
    crossover_prob, 
    mutation_prob, 
    debug_mode, 
    seed, 
    test_mode
):
    """Print parameter summary and get user confirmation."""
    print("=" * 60)
    print("OPTIMIZATION PARAMETERS SUMMARY")
    print("=" * 60)
    print(f"Optimization Type: {optimization_type}")
    print(f"Test Mode: {'Enabled' if test_mode else 'Disabled'}")
    print(f"Weather File: {weather_file}")
    print(f"Constrained Analysis: {'Enabled' if constrained_analysis else 'Disabled'}")
    if constrained_analysis:
        print(f"Overheating Criterion: {overheating_criterion}")
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



