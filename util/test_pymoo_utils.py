#!/usr/bin/env python3
"""
Unit tests for util/pymoo_utils.py covering initialization helpers,
variable selection, constraint counts, and parameter materialization.
"""

import math
import pytest

from .pymoo_utils import (
    initialize_envelope_optimization_variables,
    initialize_shading_optimization_variables,
    get_optimization_variables,
    get_problem_constraints,
    get_params,
)
from constants.shading_variables import SHADING_VARIABLE_BOUNDS


def test_initialize_envelope_optimization_variables_structure():
    vars = initialize_envelope_optimization_variables()
    # Expected keys
    for k in ['wwr_south', 'wwr_west', 'wwr_east', 'wwr_north', 'wall_construction', 'roof_construction', 'window_construction']:
        assert k in vars
    # Ranges are lists with reasonable lengths
    for k in ['wwr_south', 'wwr_west', 'wwr_east', 'wwr_north']:
        assert isinstance(vars[k], list)
        assert all(isinstance(v, int) for v in vars[k])
        assert min(vars[k]) >= 0 and max(vars[k]) <= 100
    for k in ['wall_construction', 'roof_construction', 'window_construction']:
        assert isinstance(vars[k], list)
        assert len(vars[k]) >= 1


def test_initialize_shading_optimization_variables_structure():
    vars = initialize_shading_optimization_variables()
    expected = [
        'south_overhang_depth', 'south_overhang_width', 'south_overhang_angle',
        'south_left_fin_depth', 'south_right_fin_depth',
        'west_overhang_depth', 'west_overhang_width', 'west_overhang_angle',
        'west_right_fin_depth',
    ]
    for k in expected:
        assert k in vars
        assert 'min' in vars[k] and 'max' in vars[k]
        assert vars[k]['min'] <= vars[k]['max']
    # Verify constants are the source of truth
    assert vars == SHADING_VARIABLE_BOUNDS


@pytest.mark.parametrize('opt_type, key', [
    ('shading', 'south_overhang_depth'),
    ('envelope', 'wwr_south'),
    ('shading_wwr', 'wwr_south'),
    ('SHADING', 'south_overhang_depth'),
])
def test_get_optimization_variables(opt_type, key):
    vars = get_optimization_variables(opt_type)
    assert key in vars


@pytest.mark.parametrize('constrained, criterion, expected', [
    (False, 'ASHRAE', 0),
    (False, 'CIBSE59', 0),
    (True, 'ASHRAE', 2),
    (True, 'CIBSE59', 2),
])
def test_get_problem_constraints(constrained, criterion, expected):
    assert get_problem_constraints('envelope', constrained, criterion) == expected
    assert get_problem_constraints('shading', constrained, criterion) == expected


def test_get_params_shading_builds_and_clips():
    vars = initialize_shading_optimization_variables()
    # Build x with out-of-range values to test clipping
    x = [
        -1.0,  # south_overhang_depth -> clip to min 0.0
        2.0,   # south_overhang_width -> clip to max 1.0
        90.0,  # south_overhang_angle -> clip to max 60.0
        10.0,  # south_left_fin_depth -> clip to max 5.0
        -2.0,  # south_right_fin_depth -> clip to min 0.0
        0.5,   # west_overhang_depth
        0.5,   # west_overhang_width
        30.0,  # west_overhang_angle
        1.5,   # west_right_fin_depth
    ]
    params = get_params('shading', x, vars)

    # Flat params clipped
    assert math.isclose(params['south_overhang_depth'], 0.0)
    assert math.isclose(params['south_overhang_width'], 1.0)
    assert math.isclose(params['south_overhang_angle'], 60.0)
    assert math.isclose(params['south_left_fin_depth'], 5.0)
    assert math.isclose(params['south_right_fin_depth'], 0.0)

    # Nested dims built for south and west
    assert 'shading_dimensions' in params
    dims = params['shading_dimensions']
    assert 'south' in dims and 'west' in dims
    # Overhang block present with expected keys
    for side in ['south', 'west']:
        assert 'overhang' in dims[side]
        for k in ['depth', 'width', 'angle']:
            assert k in dims[side]['overhang']
    # Left/right fins present conditionally
    assert 'left_fin' in dims['south'] and 'right_fin' in dims['south']
    assert 'right_fin' in dims['west'] and 'left_fin' not in dims['west']


def test_get_params_shading_only_west():
    # Provide only west variables
    vars = {
        'west_overhang_depth': {'min': 0.0, 'max': 2.0},
        'west_overhang_width': {'min': 0.0, 'max': 1.0},
        'west_overhang_angle': {'min': 0.0, 'max': 60.0},
        'west_right_fin_depth': {'min': 0.0, 'max': 3.0},
    }
    x = [1.0, 0.5, 15.0, 2.5]
    params = get_params('shading', x, vars)
    dims = params['shading_dimensions']
    assert set(dims.keys()) == {'west'}
    assert 'overhang' in dims['west'] and 'right_fin' in dims['west']


def test_get_params_envelope_maps_correctly_and_duplicates():
    vars = initialize_envelope_optimization_variables()
    # Select indexes within options
    # wwr_* entries are lists of ints -> returned as decimals
    # construction entries are lists of names -> returned as names
    x = [
        2,  # wwr_south -> 30% -> 0.3
        0,  # wwr_west -> 20% -> 0.2
        4,  # wwr_east -> 40% -> 0.4
        1,  # wwr_north -> 25% -> 0.25
        3,  # wall_construction -> 'Wall_4'
        2,  # roof_construction -> 'Roof_3'
        1,  # window_construction -> 'Window_2'
    ]
    params = get_params('envelope', x, vars)
    # Expect decimals derived from configured percent lists
    assert math.isclose(
        params['wwr_south'], float(vars['wwr_south'][x[0]]) / 100.0
    )
    assert math.isclose(
        params['wwr_west'], float(vars['wwr_west'][x[1]]) / 100.0
    )
    assert math.isclose(
        params['wwr_east'], float(vars['wwr_east'][x[2]]) / 100.0
    )
    assert math.isclose(
        params['wwr_north'], float(vars['wwr_north'][x[3]]) / 100.0
    )
    # Construction names should match configured lists
    assert params['wall_construction'] == vars['wall_construction'][x[4]]
    assert params['roof_construction'] == vars['roof_construction'][x[5]]
    assert params['window_construction'] == vars['window_construction'][x[6]]
    # Duplicated keys exist
    assert params['external_wall'] == params['wall_construction']
    assert params['roof'] == params['roof_construction']
    assert params['window'] == params['window_construction']


def test_get_params_shading_wwr_maps_both_sets():
    # Merge some shading bounds with WWRs
    vars = get_optimization_variables('shading_wwr')
    # Build x in the same order as vars dict iteration
    x = []
    for name, bounds in vars.items():
        # pick mid value within bounds
        val = (bounds['min'] + bounds['max']) / 2.0
        x.append(val)

    params = get_params('shading_wwr', x, vars)
    # Contains shading_dimensions and decimal WWRs
    assert 'shading_dimensions' in params
    # If keys exist, they should be floats in 0..1
    if 'wwr_south' in vars:
        assert isinstance(params.get('wwr_south'), float)
        assert 0 <= params['wwr_south'] <= 1
    if 'wwr_west' in vars:
        assert isinstance(params.get('wwr_west'), float)
        assert 0 <= params['wwr_west'] <= 1

