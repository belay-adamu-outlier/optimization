#!/usr/bin/env python3
"""
Unit tests for parametric shading helpers that create SHADING:OVERHANG and
SHADING:FIN objects, and for the overhang extension computation utility.
"""

import pytest
from unittest.mock import Mock
from types import SimpleNamespace

from .energyplus_utils import (
    create_shading_overhangs_and_fins_by_orientation,
    _compute_overhang_extensions,
)


def _make_window_vertices(x0=0.0, x1=1.0, y0=0.0, y1=0.5, z0=1.0, z1=2.5):
    """Helper to fabricate a rectangular fenestration's 4 vertices."""
    return [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z1),
        (x0, y1, z1),
    ]


class TestComputeOverhangExtensions:
    def test_explicit_extensions_preferred(self):
        verts = _make_window_vertices(x0=0.0, x1=1.0, y0=0.0, y1=1.0)
        left, right = _compute_overhang_extensions(
            verts, 'south', {'left_extension': 0.3, 'right_extension': 0.1}
        )
        assert pytest.approx(left) == 0.3
        assert pytest.approx(right) == 0.1

    def test_legacy_width_south_uses_span_x(self):
        # span_x = 1.0, rel width = 1.5 -> extra = 0.5 -> each = 0.25
        verts = _make_window_vertices(x0=0.0, x1=1.0, y0=0.0, y1=1.0)
        left, right = _compute_overhang_extensions(
            verts, 'south', {'width': 1.5}
        )
        assert pytest.approx(left) == 0.25
        assert pytest.approx(right) == 0.25

    def test_legacy_width_west_uses_span_y(self):
        # span_y = 2.0, rel width = 2.0 -> extra = 2.0 -> each = 1.0
        verts = _make_window_vertices(x0=0.0, x1=1.0, y0=0.0, y1=2.0)
        left, right = _compute_overhang_extensions(
            verts, 'west', {'width': 2.0}
        )
        assert pytest.approx(left) == 1.0
        assert pytest.approx(right) == 1.0


class TestCreateShadingOverhangsAndFins:
    def setup_method(self):
        # Mock fenestration (window)
        self.win = Mock()
        self.win.Name = 'W1'
        self.win.Building_Surface_Name = 'South:Zone1'
        # Provide vertex coordinates compatible with _get_fenestration_vertices
        verts = _make_window_vertices(x0=5.0, x1=6.0, y0=0.0, y1=1.0, z0=1.0, z1=2.5)
        for i, (x, y, z) in enumerate(verts, 1):
            setattr(self.win, f'Vertex_{i}_Xcoordinate', x)
            setattr(self.win, f'Vertex_{i}_Ycoordinate', y)
            setattr(self.win, f'Vertex_{i}_Zcoordinate', z)

        self.idf = Mock()
        self.idf.idfobjects = {
            'FENESTRATIONSURFACE:DETAILED': [self.win],
        }

        # newidfobject returns empty namespaces we can inspect later
        self.created = []

        def factory(obj_type):
            ns = SimpleNamespace(_type=obj_type)
            self.created.append(ns)
            return ns

        self.idf.newidfobject = Mock(side_effect=factory)

    def test_create_overhang_sets_all_fields(self):
        dims = {
            'south': {
                'overhang': {
                    'depth': 0.6,
                    'height_above': 0.2,
                    'left_extension_pct': 30.0,  # 30% of 1.0m -> 0.3m
                    'right_extension_pct': 10.0, # 10% of 1.0m -> 0.1m
                    'angle': 15.0,
                }
            }
        }
        create_shading_overhangs_and_fins_by_orientation(self.idf, dims)

        # One SHADING:OVERHANG created
        ovs = [o for o in self.created if o._type == 'SHADING:OVERHANG']
        assert len(ovs) == 1
        ov = ovs[0]
        assert 'Overhang' in ov.Name
        assert ov.Window_or_Door_Name == 'W1'
        assert getattr(ov, 'Base_Surface_Name', None) == 'South:Zone1'
        assert pytest.approx(ov.Depth) == 0.6
        assert pytest.approx(getattr(ov, 'Height_Above_Window_or_Door', 0.0)) == 0.2
        assert pytest.approx(getattr(ov, 'Tilt_Angle_from_Window_or_Door', 0.0)) == 15.0
        assert pytest.approx(getattr(ov, 'Left_Extension_from_Window_or_Door_Width', 0.0)) == 0.3
        assert pytest.approx(getattr(ov, 'Right_Extension_from_Window_or_Door_Width', 0.0)) == 0.1
        assert True

    def test_create_fins_set_depths_extensions_and_tilt(self):
        dims = {
            'south': {
                'left_fin': {'depth': 0.4, 'extension': 0.2, 'tilt': 10.0},
                'right_fin': {'depth': 0.5, 'extension': 0.25},
            }
        }
        create_shading_overhangs_and_fins_by_orientation(self.idf, dims)

        fins = [o for o in self.created if o._type == 'SHADING:FIN']
        assert len(fins) == 1
        fin = fins[0]
        assert 'Fin' in fin.Name
        assert fin.Window_or_Door_Name == 'W1'
        # Depths
        assert pytest.approx(fin.Left_Depth) == 0.4
        assert pytest.approx(fin.Right_Depth) == 0.5
        # Extensions
        assert pytest.approx(getattr(fin, 'Left_Extension_from_Window_or_Door', 0.0)) == 0.2
        assert pytest.approx(getattr(fin, 'Right_Extension_from_Window_or_Door', 0.0)) == 0.25
        # Tilt (single field on object)
        assert pytest.approx(getattr(fin, 'Left_Tilt_Angle_from_Window_or_Door', 0.0)) == 10.0
        assert not hasattr(fin, 'Transmittance_Schedule_Name')

    def test_both_overhang_and_fin_created(self):
        dims = {
            'south': {
                'overhang': {'depth': 0.4, 'left_extension': 0.1, 'right_extension': 0.1},
                'right_fin': {'depth': 0.3},
            }
        }
        create_shading_overhangs_and_fins_by_orientation(self.idf, dims)
        types = [o._type for o in self.created]
        assert 'SHADING:OVERHANG' in types
        assert 'SHADING:FIN' in types

    def test_missing_overhang_depth_skips_creation(self):
        dims = {'south': {'overhang': {'height_above': 0.2}}}
        create_shading_overhangs_and_fins_by_orientation(self.idf, dims)
        assert all(o._type != 'SHADING:OVERHANG' for o in self.created)

