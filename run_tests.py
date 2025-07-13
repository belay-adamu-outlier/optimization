#!/usr/bin/env python3
"""
Test runner script for the Building Envelope Optimization Project.
Provides easy ways to run tests with different options for all test files.
"""

import subprocess
import sys
import os
import glob


def get_test_files():
    """Get all test files in the current directory."""
    test_files = glob.glob("test_*.py")
    return sorted(test_files)


def run_tests_with_coverage():
    """Run all tests with coverage report."""
    print("Running all tests with coverage...")
    cmd = [
        sys.executable, "-m", "pytest", 
        "--cov=.",
        "--cov-report=term-missing",
        "--cov-report=html",
        "-v"
    ]
    subprocess.run(cmd)


def run_tests_verbose():
    """Run all tests with verbose output."""
    print("Running all tests with verbose output...")
    cmd = [
        sys.executable, "-m", "pytest", 
        "-v", 
        "-s"
    ]
    subprocess.run(cmd)


def run_tests_for_module(module_name):
    """Run tests for a specific module."""
    test_file = f"test_{module_name}.py"
    if not os.path.exists(test_file):
        print(f"Error: Test file '{test_file}' not found.")
        print(f"Available test files: {', '.join(get_test_files())}")
        return
    
    print(f"Running tests for {module_name}...")
    cmd = [
        sys.executable, "-m", "pytest", 
        test_file,
        "-v"
    ]
    subprocess.run(cmd)


def run_specific_test_class(test_class):
    """Run tests for a specific test class."""
    print(f"Running tests for {test_class}...")
    cmd = [
        sys.executable, "-m", "pytest", 
        f"-k {test_class}", 
        "-v"
    ]
    subprocess.run(cmd)


def run_specific_test(test_name):
    """Run a specific test by name."""
    print(f"Running test: {test_name}...")
    cmd = [
        sys.executable, "-m", "pytest", 
        f"-k {test_name}", 
        "-v", 
        "-s"
    ]
    subprocess.run(cmd)


def list_available_tests():
    """List all available tests."""
    test_files = get_test_files()
    
    print("Available test files:")
    for i, test_file in enumerate(test_files, 1):
        print(f"{i}. {test_file}")
    
    print("\nAvailable test classes:")
    print("Data Utils Tests:")
    print("  - (No data utilities tests currently)")
    
    print("\nEnergyPlus Utils Tests:")
    print("  - TestApplyConstructionToSurfaces")
    print("  - TestApplyInfiltrationToZones")
    print("  - TestCalculateWindowGeometry")
    print("  - TestApplyWwrToWindows")
    
    print("\nSimulation Utils Tests:")
    print("  - TestTemperatureExtraction")
    print("  - TestDistrictHeatingExtraction")
    print("  - TestSharedCacheFunctions")
    
    print("\nIDF Modification Tests:")
    print("  - (No IDF modification tests currently)")
    
    print("\nTo see all individual tests:")
    cmd = [
        sys.executable, "-m", "pytest", 
        "--collect-only", 
        "-q"
    ]
    subprocess.run(cmd)


def run_all_tests():
    """Run all tests."""
    print("Running all tests...")
    cmd = [sys.executable, "-m", "pytest", "-v"]
    subprocess.run(cmd)


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_tests.py [option]")
        print("\nOptions:")
        print("  coverage    - Run all tests with coverage report")
        print("  verbose     - Run all tests with verbose output")
        print("  list        - List all available tests")
        print("  module <name> - Run tests for specific module (e.g., data_utils, energyplus_utils)")
        print("  class <name> - Run tests for specific class")
        print("  test <name>  - Run specific test")
        print("  all         - Run all tests (default)")
        print("\nExamples:")
        print("  python run_tests.py module data_utils")
        print("  python run_tests.py class TestParameterConversion")
        print("  python run_tests.py test test_x_to_params_basic")
        return
    
    option = sys.argv[1].lower()
    
    if option == "coverage":
        run_tests_with_coverage()
    elif option == "verbose":
        run_tests_verbose()
    elif option == "list":
        list_available_tests()
    elif option == "module" and len(sys.argv) > 2:
        run_tests_for_module(sys.argv[2])
    elif option == "class" and len(sys.argv) > 2:
        run_specific_test_class(sys.argv[2])
    elif option == "test" and len(sys.argv) > 2:
        run_specific_test(sys.argv[2])
    elif option == "all":
        run_all_tests()
    else:
        print("Invalid option. Use 'python run_tests.py' for help.")


if __name__ == "__main__":
    main() 