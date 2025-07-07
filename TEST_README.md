# Test Suite for energyplus_utils.py

This directory contains comprehensive test scenarios for the top 4 functions in `energyplus_utils.py` using pytest.

## Test Coverage

The test suite covers the following functions:

1. **`apply_construction_to_surfaces`** - Tests applying different construction types to building surfaces
2. **`apply_infiltration_to_zones`** - Tests applying infiltration rates to zones with proper unit conversion
3. **`calculate_window_geometry`** - Tests window geometry calculations with various aspect ratios
4. **`apply_wwr_to_windows`** - Tests window-to-wall ratio applications based on wall orientation

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

# Run tests for a specific class
python run_tests.py class TestApplyConstructionToSurfaces

# Run a specific test
python run_tests.py test test_apply_wall_construction
```

### Using pytest Directly

```bash
# Run all tests
pytest test_run_energyplus.py -v

# Run tests with coverage
pytest test_run_energyplus.py --cov=energyplus_utils --cov-report=html -v

# Run specific test class
pytest test_run_energyplus.py::TestApplyConstructionToSurfaces -v

# Run specific test
pytest test_run_energyplus.py::TestApplyConstructionToSurfaces::test_apply_wall_construction -v
```

## Test Scenarios

### TestApplyConstructionToSurfaces
- ✅ Applying construction to wall surfaces
- ✅ Applying construction to roof surfaces  
- ✅ Applying construction to window surfaces
- ✅ Handling empty surface lists
- ✅ Filtering outdoor vs indoor walls

### TestApplyInfiltrationToZones
- ✅ Applying infiltration to zones with existing infiltration objects
- ✅ Creating new infiltration objects when none exist
- ✅ Proper unit conversion from 50 Pa to ambient conditions
- ✅ Handling empty zone lists

### TestCalculateWindowGeometry
- ✅ X-fixed axis window geometry calculation
- ✅ Y-fixed axis window geometry calculation
- ✅ Aspect ratio selection based on window area
- ✅ Error handling for non-axis-aligned windows
- ✅ Window centering verification
- ✅ Sill height preservation

### TestApplyWwrToWindows
- ✅ Applying WWR to east-facing walls
- ✅ Applying WWR to south-facing walls
- ✅ Applying WWR to west-facing walls
- ✅ Applying WWR to north-facing walls
- ✅ Handling missing parent walls
- ✅ Handling incomplete vertex data
- ✅ Multiple fenestration support
- ✅ Azimuth boundary condition testing

## Test Features

- **Mocking**: Uses `unittest.mock` to mock IDF objects and avoid actual EnergyPlus dependencies
- **Edge Cases**: Tests handle edge cases like empty lists, missing data, and boundary conditions
- **Comprehensive Coverage**: Tests cover normal operation, error conditions, and edge cases
- **Isolation**: Each test is independent and uses fresh mock objects

## Expected Test Results

When running the tests, you should see output similar to:

```
test_run_energyplus.py::TestApplyConstructionToSurfaces::test_apply_wall_construction PASSED
test_run_energyplus.py::TestApplyConstructionToSurfaces::test_apply_roof_construction PASSED
test_run_energyplus.py::TestApplyConstructionToSurfaces::test_apply_window_construction PASSED
...
```

All tests should pass, indicating that the functions work correctly under various scenarios.

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
3. Verify that `energyplus_utils.py` is in the same directory as the test files
4. Check that the function signatures in the tests match the actual functions 