import matplotlib
matplotlib.use('Agg')

import pytest


def _make_shared_cache():
    # Two entries in generation 1, one entry in generation 2
    return {
        'k1': {
            'objectives': [100.0, 20.0],
            'generation': 1,
            'params': {
                'external_wall': 'R17 - Wall',
                'roof': 'R18 - Roof',
                'window': 'Dbl LoE Spec'
            }
        },
        'k2': {
            'objectives': [110.0, 22.0],
            'generation': 1,
            'params': {
                'external_wall': 'R17 - Wall',
                'roof': 'R18 - Roof',
                'window': 'Trp LoE'
            }
        },
        'k3': {
            'objectives': [90.0, 18.0],
            'generation': 2,
            'params': {
                'external_wall': 'R17 - Wall',
                'roof': 'R18 - Roof',
                'window': 'Trp LoE Film'
            }
        }
    }


class _DummyPop:
    def get(self, name):
        return [1, 2, 3]


class _DummyProblem:
    def __init__(self):
        self._gen = 0

    def set_generation(self, g):
        self._gen = g


class _DummyAlgorithm:
    def __init__(self):
        self.pop = _DummyPop()
        self.problem = _DummyProblem()


def test_notify_appends_only_current_generation(monkeypatch):
    from config.optimization_config import EvaluationCallback

    # Prepare shared cache and callback
    shared_cache = _make_shared_cache()
    optimization_variables = {
        'external_wall': ['R17 - Wall'],
        'roof': ['R18 - Roof'],
        'window': ['Dbl LoE Spec']
    }

    # Capture generations and counts passed to save_to_csv
    calls = []

    def fake_save_to_csv(csv_filename,
                         solutions_by_generation,
                         solutions_by_key,
                         variable_names,
                         generation,
                         problem_type='envelope',
                         constrained_analysis=False,
                         overheating_criterion='ASHRAE',
                         objective_names=None,
                         multiobjective=True):
        count = len(solutions_by_generation.get(generation, []))
        calls.append((generation, count))

    # No-op plotting during test
    def noop_update(*args, **kwargs):
        return None

    monkeypatch.setattr('config.optimization_config.save_to_csv', fake_save_to_csv)
    monkeypatch.setattr('config.optimization_config.update_envelope_optimization_plot', noop_update)

    cb = EvaluationCallback(
        optimization_variables,
        shared_cache=shared_cache,
        debug_mode=True,
        optimization_type='envelope',
        constrained_analysis=False,
        overheating_criterion='ASHRAE'
    )

    algo = _DummyAlgorithm()

    # First generation -> should save only gen 1 rows (2 entries)
    cb.notify(algo)
    # Second generation -> should save only gen 2 rows (1 entry)
    cb.notify(algo)

    assert calls[0] == (1, 2)
    assert calls[1] == (2, 1)


def test_rebuild_solutions_includes_max_overheating_zones(monkeypatch):
    from config.optimization_config import EvaluationCallback

    shared_cache = {
        'k1': {
            'objectives': [10.0, 20.0],
            'generation': 1,
            'max_overheating_hours': 55.0,
            'max_overheating_zone': 'NE:BR',
            'max_living_room_overheating': 40.0,
            'max_bedroom_overheating': 55.0,
            'max_living_room_overheating_zone': 'SW:LR',
            'max_bedroom_overheating_zone': 'NE:BR',
            'feasible': True,
            'params': {
                'external_wall': 'R17 - Wall',
                'roof': 'R18 - Roof',
                'window': 'Dbl LoE Spec'
            }
        }
    }
    optimization_variables = {
        'external_wall': ['R17 - Wall'],
        'roof': ['R18 - Roof'],
        'window': ['Dbl LoE Spec']
    }

    monkeypatch.setattr(
        'config.optimization_config.initialize_csv',
        lambda *args, **kwargs: None
    )

    cb = EvaluationCallback(
        optimization_variables,
        shared_cache=shared_cache,
        debug_mode=True,
        optimization_type='envelope',
        constrained_analysis=True,
        overheating_criterion='CIBSE59'
    )

    cb._rebuild_solutions_from_cache()

    solution = cb.solutions_by_key['k1']
    assert solution['max_living_room_overheating_zone'] == 'SW:LR'
    assert solution['max_bedroom_overheating_zone'] == 'NE:BR'
    assert solution['max_overheating_zone'] == 'NE:BR'


def test_rebuild_solutions_copies_zones_when_unconstrained(monkeypatch):
    from config.optimization_config import EvaluationCallback

    shared_cache = {
        'k1': {
            'objectives': [10.0, 20.0],
            'generation': 1,
            'max_living_room_overheating': 40.0,
            'max_bedroom_overheating': 55.0,
            'max_living_room_overheating_zone': 'N:LR',
            'max_bedroom_overheating_zone': 'SE:BR',
            'params': {
                'external_wall': 'R17 - Wall',
                'roof': 'R18 - Roof',
                'window': 'Dbl LoE Spec'
            }
        }
    }
    optimization_variables = {
        'external_wall': ['R17 - Wall'],
        'roof': ['R18 - Roof'],
        'window': ['Dbl LoE Spec']
    }

    monkeypatch.setattr(
        'config.optimization_config.initialize_csv',
        lambda *args, **kwargs: None
    )

    cb = EvaluationCallback(
        optimization_variables,
        shared_cache=shared_cache,
        debug_mode=True,
        optimization_type='envelope',
        constrained_analysis=False,
        overheating_criterion='CIBSE59'
    )

    cb._rebuild_solutions_from_cache()

    solution = cb.solutions_by_key['k1']
    assert solution['max_living_room_overheating_zone'] == 'N:LR'
    assert solution['max_bedroom_overheating_zone'] == 'SE:BR'


