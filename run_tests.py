#!/usr/bin/env python3
"""
Test runner script for energyplus_utils.py tests.
Provides easy ways to run tests with different options.
"""

import subprocess
import sys
import os


def run_tests_with_coverage():
    """Run tests with coverage report."""
    print("Running tests with coverage...")
    cmd = [
        sys.executable, "-m", "pytest", 
        "test_energyplus_utils.py", 
        "--cov=energyplus_utils",
        "--cov-report=term-missing",
        "--cov-report=html",
        "-v"
    ]
    subprocess.run(cmd)


def run_tests_verbose():
    """Run tests with verbose output."""
    print("Running tests with verbose output...")
    cmd = [
        sys.executable, "-m", "pytest", 
        "test_energyplus_utils.py", 
        "-v", 
        "-s"
    ]
    subprocess.run(cmd)


def run_specific_test_class(test_class):
    """Run tests for a specific test class."""
    print(f"Running tests for {test_class}...")
    cmd = [
        sys.executable, "-m", "pytest", 
        f"test_energyplus_utils.py::{test_class}", 
        "-v"
    ]
    subprocess.run(cmd)


def run_specific_test(test_name):
    """Run a specific test by name."""
    print(f"Running test: {test_name}...")
    cmd = [
        sys.executable, "-m", "pytest", 
        f"test_energyplus_utils.py::{test_name}", 
        "-v", 
        "-s"
    ]
    subprocess.run(cmd)


def list_available_tests():
    """List all available tests."""
    print("Available test classes:")
    print("1. TestApplyConstructionToSurfaces")
    print("2. TestApplyInfiltrationToZones")
    print("3. TestCalculateWindowGeometry")
    print("4. TestApplyWwrToWindows")
    print("\nAvailable individual tests:")
    cmd = [
        sys.executable, "-m", "pytest", 
        "test_energyplus_utils.py", 
        "--collect-only", 
        "-q"
    ]
    subprocess.run(cmd)


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_tests.py [option]")
        print("\nOptions:")
        print("  coverage    - Run tests with coverage report")
        print("  verbose     - Run tests with verbose output")
        print("  list        - List all available tests")
        print("  class <name> - Run tests for specific class")
        print("  test <name>  - Run specific test")
        print("  all         - Run all tests (default)")
        return
    
    option = sys.argv[1].lower()
    
    if option == "coverage":
        run_tests_with_coverage()
    elif option == "verbose":
        run_tests_verbose()
    elif option == "list":
        list_available_tests()
    elif option == "class" and len(sys.argv) > 2:
        run_specific_test_class(sys.argv[2])
    elif option == "test" and len(sys.argv) > 2:
        run_specific_test(sys.argv[2])
    elif option == "all":
        print("Running all tests...")
        cmd = [sys.executable, "-m", "pytest", "test_energyplus_utils.py", "-v"]
        subprocess.run(cmd)
    else:
        print("Invalid option. Use 'python run_tests.py' for help.")


if __name__ == "__main__":
    main()
