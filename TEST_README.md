# Test Suite for Building Envelope Optimization Project

This directory contains comprehensive test scenarios for the refactored building envelope optimization project using pytest.



## Test Coverage

The test suite covers the following modules:

### 1. Data Utilities Tests

Currently no dedicated test file for `util/data_utils.py` (removed to simplify test suite).

### 2. `util/test_energyplus_utils.py` - EnergyPlus Utilities Tests

Tests for the core functions in `util/energyplus_utils.py`:

- **`apply_construction_to_surfaces`** - Tests applying different construction types to building surfaces
- **`apply_infiltration_to_zones`** - Tests applying infiltration rates to zones with proper unit conversion
- **`calculate_window_geometry`** - Tests window geometry calculations with various aspect ratios
- **`apply_wwr_to_windows`** - Tests window-to-wall ratio applications based on wall orientation
- **`rename_outside_walls_by_orientation`** - Tests renaming exterior walls based on geometric orientation
- **`create_shading_elements_by_orientation`** - Tests creating shading elements (overhangs, fins) for different facades
- **`modify_idf`** - Tests high-level IDF modification with various envelope parameters

### 3. `util/test_simulation_utils.py` - Simulation Utilities Tests

Comprehensive tests for all functions in `util/simulation_utils.py`:

- **Temperature Extraction**: `extract_zone_temperatures()` - Converting temperature data to DataFrames
- **District Heating Extraction**: `extract_district_heating_from_htm()` - Extracting heating values from HTML files
- **Shared Cache Management**: `create_shared_cache()`, `make_cache_key()`, `set_in_cache()`, `get_from_cache()`, etc. - Multiprocessing cache operations
- **Cache File Saving**: `save_cache_to_file()` - Updates existing cache file with new entries rather than creating new files each generation. File writing and exception handling are fully tested via mocking. Cache now stores parameters, objectives, and generation info for complete solution reconstruction.

### 4. Root Directory Tests

- **`test_overheating_analysis.py`** - Tests for overheating analysis functionality
- **`simple_test.py`** - Simple utility function tests
- **`debug_test.py`** - Debug and verification tests

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
# Run all tests (including util/ folder)
pytest -v

# Run tests with coverage
pytest --cov=. --cov-report=html -v

# Run specific test file
pytest util/test_energyplus_utils.py -v
pytest util/test_simulation_utils.py -v
pytest test_overheating_analysis.py -v

# Run specific test class
pytest util/test_energyplus_utils.py::TestApplyConstructionToSurfaces -v
pytest util/test_simulation_utils.py::TestTemperatureExtraction -v

# Run specific test
pytest util/test_energyplus_utils.py::TestApplyConstructionToSurfaces::test_apply_wall_construction -v
pytest util/test_simulation_utils.py::TestTemperatureExtraction::test_extract_zone_temperatures_basic -v

# Run only tests in util/ folder
pytest util/ -v

# Run only tests in root directory
pytest test_overheating_analysis.py simple_test.py debug_test.py -v
```

## Test Scenarios

### util/test_simulation_utils.py

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
- ✅ Cache file saving (updates existing file with new entries, file writing and error handling, including exceptions, are fully tested)

### util/test_energyplus_utils.py

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

#### TestRenameOutsideWallsByOrientation

- ✅ Renaming outside walls according to orientation (North, South, East, West)
- ✅ Handling multiple zones with different wall configurations
- ✅ Processing zones with no outdoor walls
- ✅ Processing empty zones gracefully
- ✅ Proper wall classification based on coordinate geometry
- ✅ Fenestration reference updates for renamed walls

#### TestCreateShadingElementsByOrientation

- ✅ Creating shading elements for different orientations (south, east, west)
- ✅ Handling missing dimensions gracefully
- ✅ Processing fenestrations with no matching orientations
- ✅ Creating elements with empty dimensions dictionary
- ✅ Handling None dimensions parameter
- ✅ Conditional creation of overhangs, fins, and right/left fins

#### TestModifyIdf

- ✅ Modifying IDF with all envelope parameters
- ✅ Processing partial parameter sets
- ✅ Save functionality with default and custom paths
- ✅ Handling empty and None parameter dictionaries
- ✅ WWR-only parameter processing
- ✅ Albedo-only parameter processing
- ✅ Conditional application of solar absorptance
- ✅ Integration with all envelope modification functions

### test_overheating_analysis.py

#### TestOverheatingAnalysis

- ✅ ASHRAE adaptive threshold calculations
- ✅ CIBSE59 running mean calculations
- ✅ Overheating hours calculation
- ✅ Zone type detection
- ✅ Bedroom sleeping hours calculation
- ✅ Living room temperature difference calculations
- ✅ Custom summer months handling
- ✅ Empty data handling
- ✅ Missing outdoor data handling

## Test Features

- **Mocking**: Uses `unittest.mock` to mock file operations, plotting, and EnergyPlus dependencies
- **Edge Cases**: Tests handle edge cases like empty data, missing files, and boundary conditions
- **Comprehensive Coverage**: Tests cover normal operation, error conditions, and edge cases
- **Isolation**: Each test is independent and uses fresh mock objects
- **File Backup**: Tests include proper file backup handling for CSV operations
- **Plot Mocking**: Uses matplotlib mocking to avoid GUI dependencies during testing
- **File Writing**: File writing and error handling for cache saving are robustly tested using mocks
- **Single Source of Truth**: All operations (CSV, plotting, cache file) use the same shared cache to ensure consistency
- **Package Structure**: Tests in `util/` folder use relative imports (e.g., `from .energyplus_utils`) while root tests use absolute imports

## Test Statistics

The test suite currently contains **76 tests** covering:
- **Building envelope functions**: 37 tests
- **Shading schedule functions**: 12 tests
- **Shading optimization helpers**: 8 tests
- **Simulation utilities**: 14 tests
- **Overheating analysis**: 9 tests

## Expected Test Results

When running the tests, you should see output similar to:

```
util/test_simulation_utils.py::TestTemperatureExtraction::test_extract_zone_temperatures_basic PASSED
util/test_simulation_utils.py::TestDistrictHeatingExtraction::test_extract_district_heating_basic PASSED
util/test_simulation_utils.py::TestSharedCacheFunctions::test_create_shared_cache PASSED
...
util/test_energyplus_utils.py::TestApplyConstructionToSurfaces::test_apply_wall_construction PASSED
util/test_energyplus_utils.py::TestApplyConstructionToSurfaces::test_apply_roof_construction PASSED
...
test_overheating_analysis.py::TestOverheatingAnalysis::test_analyze_overheating_with_ASHRAE PASSED
...
```

All tests should pass, indicating that the refactored functions work correctly under various scenarios.



## Adding New Tests

1. **For utility functions**: Add test methods to the appropriate test class in the `util/` folder
2. **For main application functions**: Add test methods to test classes in the root directory
3. Follow the naming convention: `test_<function_name>_<scenario>`
4. Use descriptive docstrings explaining what the test validates
5. Use appropriate assertions to verify expected behavior
6. Run the tests to ensure they pass

## Test Coverage Standards

All functions in the test suite have comprehensive test coverage including:

- **Happy path scenarios**: Normal operation with valid parameters
- **Edge cases**: Empty parameters, None values, missing data
- **Error handling**: Graceful degradation when expected data is missing
- **Integration testing**: Functions work together correctly
- **Mock testing**: Proper isolation from external dependencies
- **Parameter validation**: Various parameter combinations and partial sets

All functions follow the project's testing standards and include proper docstrings for maintainability.

## Troubleshooting

1. Check that all dependencies are installed: `pip install -r test_requirements.txt`
2. Ensure you're in the correct directory (project root)
3. Verify that all module files are in their correct locations (`util/` folder for utilities)
4. Check that the function signatures in the tests match the actual functions
5. For file operation tests, ensure proper file permissions and backup handling
6. For tests in `util/` folder, ensure relative imports are used correctly
7. For tests in root directory, ensure absolute imports are used correctly 