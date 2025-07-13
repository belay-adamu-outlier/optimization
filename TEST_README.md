# Test Suite for Building Envelope Optimization Project

This directory contains comprehensive test scenarios for the refactored building envelope optimization project using pytest.

## Test Coverage

The test suite covers the following modules:

### 1. Data Utilities Tests

Currently no dedicated test file for data_utils.py (removed to simplify test suite).

### 2. `test_energyplus_utils.py` - EnergyPlus Utilities Tests

Tests for the top 4 functions in `energyplus_utils.py`:

- **`apply_construction_to_surfaces`** - Tests applying different construction types to building surfaces
- **`apply_infiltration_to_zones`** - Tests applying infiltration rates to zones with proper unit conversion
- **`calculate_window_geometry`** - Tests window geometry calculations with various aspect ratios
- **`apply_wwr_to_windows`** - Tests window-to-wall ratio applications based on wall orientation

### 3. `test_simulation_utils.py` - Simulation Utilities Tests

Comprehensive tests for all functions in `simulation_utils.py`:

- **Temperature Extraction**: `extract_zone_temperatures()` - Converting temperature data to DataFrames
- **District Heating Extraction**: `extract_district_heating_from_htm()` - Extracting heating values from HTML files
- **Shared Cache Management**: `create_shared_cache()`, `make_cache_key()`, `set_in_cache()`, `get_from_cache()`, etc. - Multiprocessing cache operations
- **Cache File Saving**: `save_cache_to_file()` - File writing and exception handling are now fully tested via mocking

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
pytest test_energyplus_utils.py -v
pytest test_simulation_utils.py -v

# Run specific test class
pytest test_energyplus_utils.py::TestApplyConstructionToSurfaces -v
pytest test_simulation_utils.py::TestTemperatureExtraction -v

# Run specific test
pytest test_energyplus_utils.py::TestApplyConstructionToSurfaces::test_apply_wall_construction -v
pytest test_simulation_utils.py::TestTemperatureExtraction::test_extract_zone_temperatures_basic -v
```

## Test Scenarios

### test_simulation_utils.py

#### TestTemperatureExtraction

- ✅ Basic zone temperature extraction with outdoor data
- ✅ Zone temperature extraction without outdoor data
- ✅ Empty temperature data handling

#### TestDistrictHeatingExtraction

- ✅ Basic district heating extraction from HTML files
- ✅ File not found error handling
- ✅ No matching data error handling
- ✅ Invalid numeric value error handling

#### TestSharedCacheFunctions

- ✅ Shared cache creation
- ✅ Cache key generation
- ✅ Basic cache setting and retrieval
- ✅ Cache with generation info
- ✅ Cache entry retrieval
- ✅ Generation retrieval from cache
- ✅ Old format cache compatibility
- ✅ Non-existent key handling
- ✅ Cache file saving (file writing and error handling, including exceptions, are fully tested)

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
- **File Writing**: File writing and error handling for cache saving are robustly tested using mocks

## Expected Test Results

When running the tests, you should see output similar to:

```
test_simulation_utils.py::TestTemperatureExtraction::test_extract_zone_temperatures_basic PASSED
test_simulation_utils.py::TestDistrictHeatingExtraction::test_extract_district_heating_basic PASSED
test_simulation_utils.py::TestSharedCacheFunctions::test_create_shared_cache PASSED
...
test_energyplus_utils.py::TestApplyConstructionToSurfaces::test_apply_wall_construction PASSED
test_energyplus_utils.py::TestApplyConstructionToSurfaces::test_apply_roof_construction PASSED
...
```

All tests should pass, indicating that the refactored functions work correctly under various scenarios.

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