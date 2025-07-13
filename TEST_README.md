# Test Suite for Building Envelope Optimization Project

This directory contains comprehensive test scenarios for the refactored building envelope optimization project using pytest.

## Project Structure

The project has been refactored for better modularity and maintainability:

- **`main.py`** - Main optimization logic with NSGA-II algorithm
- **`data_utils.py`** - Data processing, CSV operations, plotting, and utility functions
- **`simulation_utils.py`** - Simulation caching and management utilities
- **`energyplus_utils.py`** - EnergyPlus simulation and IDF manipulation functions
- **`cli_utils.py`** - Command-line interface and user interaction utilities
- **`overheating_analysis.py`** - Overheating analysis and calculation functions

## Test Coverage

The test suite covers the following modules:

### 1. `test_data_utils.py` - Data Utilities Tests
Comprehensive tests for all functions in `data_utils.py`:

- **Parameter Conversion**: `x_to_params()` - Converting optimization variables to parameter dictionaries
- **Table Printing**: `print_optimization_table()` - Displaying optimization results in formatted tables
- **Pareto Optimality**: `update_pareto_optimality_for_display()` - Calculating Pareto optimal solutions
- **CSV Operations**: `initialize_csv()`, `save_to_csv()`, `save_final_results()` - File I/O operations
- **Plotting**: `initialize_plot()`, `update_plot()` - Real-time plotting during optimization
- **Configuration**: `load_optimization_variables()`, `load_config()` - Loading project configuration

### 2. `test_energyplus_utils.py` - EnergyPlus Utilities Tests
Tests for the top 4 functions in `energyplus_utils.py`:

- **`apply_construction_to_surfaces`** - Tests applying different construction types to building surfaces
- **`apply_infiltration_to_zones`** - Tests applying infiltration rates to zones with proper unit conversion
- **`calculate_window_geometry`** - Tests window geometry calculations with various aspect ratios
- **`apply_wwr_to_windows`** - Tests window-to-wall ratio applications based on wall orientation

### 3. `test_modify_idf_function.py` - IDF Modification Tests
Tests for IDF file modification functionality.

## Installation

1. Install test dependencies:
```bash
pip install -r test_requirements.txt
```

## Running Tests

### Using the Test Runner Script

The `run_tests.py` script provides easy ways to run tests:

```bash
# Run all tests
python run_tests.py all

# Run tests with coverage report
python run_tests.py coverage

# Run tests with verbose output
python run_tests.py verbose

# List all available tests
python run_tests.py list

# Run tests for a specific module
python run_tests.py module data_utils

# Run tests for a specific class
python run_tests.py class TestParameterConversion

# Run a specific test
python run_tests.py test test_x_to_params_basic
```

### Using pytest Directly

```bash
# Run all tests
pytest -v

# Run tests with coverage
pytest --cov=. --cov-report=html -v

# Run specific test file
pytest test_data_utils.py -v
pytest test_energyplus_utils.py -v
pytest test_modify_idf_function.py -v

# Run specific test class
pytest test_data_utils.py::TestParameterConversion -v
pytest test_energyplus_utils.py::TestApplyConstructionToSurfaces -v

# Run specific test
pytest test_data_utils.py::TestParameterConversion::test_x_to_params_basic -v
pytest test_energyplus_utils.py::TestApplyConstructionToSurfaces::test_apply_wall_construction -v
```

## Test Scenarios

### test_data_utils.py

#### TestParameterConversion
- ✅ Basic parameter conversion with valid inputs
- ✅ Handling empty optimization variables
- ✅ Edge case with single variable
- ✅ Large number of variables
- ✅ Invalid index handling

#### TestTablePrinting
- ✅ Basic table printing with solutions
- ✅ Empty solutions handling
- ✅ Single solution display
- ✅ Large dataset handling
- ✅ Missing variable handling

#### TestParetoOptimality
- ✅ Basic Pareto optimality calculation
- ✅ Single solution (always Pareto optimal)
- ✅ No Pareto optimal solutions
- ✅ All solutions Pareto optimal
- ✅ Edge case with identical objectives

#### TestCSVOperations
- ✅ CSV file initialization
- ✅ Saving data to CSV
- ✅ Final results saving with metadata
- ✅ File backup handling
- ✅ Error handling for file operations

#### TestPlotting
- ✅ Plot initialization
- ✅ Plot updates with new data
- ✅ Empty data handling
- ✅ Large dataset plotting
- ✅ Plot customization

#### TestConfiguration
- ✅ Loading optimization variables
- ✅ Loading configuration
- ✅ Error handling for missing files
- ✅ Invalid JSON handling

### test_energyplus_utils.py

#### TestApplyConstructionToSurfaces
- ✅ Applying construction to wall surfaces
- ✅ Applying construction to roof surfaces  
- ✅ Applying construction to window surfaces
- ✅ Handling empty surface lists
- ✅ Filtering outdoor vs indoor walls

#### TestApplyInfiltrationToZones
- ✅ Applying infiltration to zones with existing infiltration objects
- ✅ Creating new infiltration objects when none exist
- ✅ Proper unit conversion from 50 Pa to ambient conditions
- ✅ Handling empty zone lists

#### TestCalculateWindowGeometry
- ✅ X-fixed axis window geometry calculation
- ✅ Y-fixed axis window geometry calculation
- ✅ Aspect ratio selection based on window area
- ✅ Error handling for non-axis-aligned windows
- ✅ Window centering verification
- ✅ Sill height preservation

#### TestApplyWwrToWindows
- ✅ Applying WWR to east-facing walls
- ✅ Applying WWR to south-facing walls
- ✅ Applying WWR to west-facing walls
- ✅ Applying WWR to north-facing walls
- ✅ Handling missing parent walls
- ✅ Handling incomplete vertex data
- ✅ Multiple fenestration support
- ✅ Azimuth boundary condition testing

## Test Features

- **Mocking**: Uses `unittest.mock` to mock file operations, plotting, and EnergyPlus dependencies
- **Edge Cases**: Tests handle edge cases like empty data, missing files, and boundary conditions
- **Comprehensive Coverage**: Tests cover normal operation, error conditions, and edge cases
- **Isolation**: Each test is independent and uses fresh mock objects
- **File Backup**: Tests include proper file backup handling for CSV operations
- **Plot Mocking**: Uses matplotlib mocking to avoid GUI dependencies during testing

## Expected Test Results

When running the tests, you should see output similar to:

```
test_data_utils.py::TestParameterConversion::test_x_to_params_basic PASSED
test_data_utils.py::TestParameterConversion::test_x_to_params_empty PASSED
test_data_utils.py::TestTablePrinting::test_print_optimization_table_basic PASSED
...
test_energyplus_utils.py::TestApplyConstructionToSurfaces::test_apply_wall_construction PASSED
test_energyplus_utils.py::TestApplyConstructionToSurfaces::test_apply_roof_construction PASSED
test_modify_idf_function.py::TestModifyIdfFunction::test_modify_idf_function PASSED
...
```

All tests should pass, indicating that the refactored functions work correctly under various scenarios.

## Refactoring Benefits

The refactoring process has provided several benefits:

1. **Modularity**: Functions are now organized by purpose in separate modules
2. **Maintainability**: Easier to locate and modify specific functionality
3. **Testability**: Each module can be tested independently
4. **Readability**: Main optimization logic is cleaner and more focused
5. **Reusability**: Utility functions can be easily imported and used elsewhere

## Adding New Tests

To add new tests:

1. Add test methods to the appropriate test class
2. Follow the naming convention: `test_<function_name>_<scenario>`
3. Use descriptive docstrings explaining what the test validates
4. Use appropriate assertions to verify expected behavior
5. Run the tests to ensure they pass

## Troubleshooting

If tests fail:

1. Check that all dependencies are installed: `pip install -r test_requirements.txt`
2. Ensure you're in the correct directory
3. Verify that all module files are in the same directory as the test files
4. Check that the function signatures in the tests match the actual functions
5. For file operation tests, ensure proper file permissions and backup handling 