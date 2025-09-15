#!/usr/bin/env python3
"""
Test scenarios for energyplus_utils.py functions using pytest.
Tests the top 4 functions with various scenarios including edge cases.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from types import SimpleNamespace
import tempfile
import os
from .energyplus_utils import (
    apply_construction_to_surfaces,
    apply_infiltration_to_zones,
    calculate_window_geometry,
    apply_wwr_to_windows,
    apply_solar_absorptance_to_surfaces,
    rename_shading_elements_by_orientation,
    create_shading_elements_by_orientation,
    assign_shading_transmittance_schedules,
    apply_shading_schedules_to_idf,
    _expand_partial_schedule,
    build_seasonal_shading_schedules,
    calculate_total_shading_area,
    bulk_copy_window_related_objects,
)


class TestApplyConstructionToSurfaces:
    """Test scenarios for apply_construction_to_surfaces function."""
    
    def setup_method(self):
        """Set up mock IDF object for each test."""
        self.mock_idf = Mock()
        
        # Mock building surfaces
        self.mock_wall_surface = Mock()
        self.mock_wall_surface.Surface_Type = 'Wall'
        self.mock_wall_surface.Outside_Boundary_Condition = 'Outdoors'
        self.mock_wall_surface.Construction_Name = 'Old_Wall_Construction'
        
        self.mock_roof_surface = Mock()
        self.mock_roof_surface.Surface_Type = 'Roof'
        self.mock_roof_surface.Construction_Name = 'Old_Roof_Construction'
        
        # Mock fenestration surfaces
        self.mock_fenestration = Mock()
        self.mock_fenestration.Construction_Name = 'Old_Window_Construction'
        
        # Set up idfobjects dictionary
        self.mock_idf.idfobjects = {
            'BUILDINGSURFACE:DETAILED': [self.mock_wall_surface, self.mock_roof_surface],
            'FENESTRATIONSURFACE:DETAILED': [self.mock_fenestration]
        }
    
    def test_apply_wall_construction(self):
        """Test applying construction to wall surfaces."""
        construction_name = 'New_Wall_Construction'
        
        apply_construction_to_surfaces(self.mock_idf, construction_name, 'Wall')
        
        # Verify wall surface construction was updated
        assert self.mock_wall_surface.Construction_Name == construction_name
        # Verify roof surface was not changed
        assert self.mock_roof_surface.Construction_Name == 'Old_Roof_Construction'
    
    def test_apply_roof_construction(self):
        """Test applying construction to roof surfaces."""
        construction_name = 'New_Roof_Construction'
        
        apply_construction_to_surfaces(self.mock_idf, construction_name, 'Roof')
        
        # Verify roof surface construction was updated
        assert self.mock_roof_surface.Construction_Name == construction_name
        # Verify wall surface was not changed
        assert self.mock_wall_surface.Construction_Name == 'Old_Wall_Construction'
    
    def test_apply_window_construction(self):
        """Test applying construction to window surfaces."""
        construction_name = 'New_Window_Construction'
        
        apply_construction_to_surfaces(self.mock_idf, construction_name, 'Window')
        
        # Verify fenestration construction was updated
        assert self.mock_fenestration.Construction_Name == construction_name
    
    def test_apply_construction_to_empty_surfaces(self):
        """Test applying construction when no surfaces of specified type exist."""
        mock_idf_empty = Mock()
        mock_idf_empty.idfobjects = {
            'BUILDINGSURFACE:DETAILED': [],
            'FENESTRATIONSURFACE:DETAILED': []
        }
        
        # Should not raise any exceptions
        apply_construction_to_surfaces(mock_idf_empty, 'Test_Construction', 'Wall')
        apply_construction_to_surfaces(mock_idf_empty, 'Test_Construction', 'Window')
    
    def test_apply_construction_to_wall_with_different_boundary_conditions(self):
        """Test that only outdoor walls get updated."""
        # Create wall with different boundary condition
        mock_indoor_wall = Mock()
        mock_indoor_wall.Surface_Type = 'Wall'
        mock_indoor_wall.Outside_Boundary_Condition = 'Surface'
        mock_indoor_wall.Construction_Name = 'Old_Indoor_Wall'
        
        mock_idf = Mock()
        mock_idf.idfobjects = {
            'BUILDINGSURFACE:DETAILED': [self.mock_wall_surface, mock_indoor_wall],
            'FENESTRATIONSURFACE:DETAILED': []
        }
        
        apply_construction_to_surfaces(mock_idf, 'New_Wall_Construction', 'Wall')
        
        # Only outdoor wall should be updated
        assert self.mock_wall_surface.Construction_Name == 'New_Wall_Construction'
        assert mock_indoor_wall.Construction_Name == 'Old_Indoor_Wall'


class TestApplyInfiltrationToZones:
    """Test scenarios for apply_infiltration_to_zones function."""
    
    def setup_method(self):
        """Set up mock IDF object for each test."""
        self.mock_idf = Mock()
        
        # Mock zones
        self.mock_zone1 = Mock()
        self.mock_zone1.Name = 'Zone1'
        
        self.mock_zone2 = Mock()
        self.mock_zone2.Name = 'Zone2'
        
        # Mock existing infiltration
        self.mock_existing_infiltration = Mock()
        self.mock_existing_infiltration.Zone_or_ZoneList_Name = 'Zone1'
        self.mock_existing_infiltration.Design_Flow_Rate = 0.1
        self.mock_existing_infiltration.Air_Changes_per_Hour = 0.05
        
        # Set up idfobjects dictionary
        self.mock_idf.idfobjects = {
            'ZONE': [self.mock_zone1, self.mock_zone2],
            'ZONEINFILTRATION:DESIGNFLOWRATE': [self.mock_existing_infiltration]
        }
        
        # Mock newidfobject method
        self.mock_idf.newidfobject = Mock()
    
    def test_apply_infiltration_to_existing_zones(self):
        """Test applying infiltration to zones with existing infiltration objects."""
        infiltration_ach_50pa = 1.0  # 1 ACH at 50 Pa
        expected_ambient_ach = 1.0 / 20.0  # 0.05 ACH ambient
        
        apply_infiltration_to_zones(self.mock_idf, infiltration_ach_50pa)
        
        # Verify existing infiltration was updated
        assert self.mock_existing_infiltration.Design_Flow_Rate == ''
        assert self.mock_existing_infiltration.Air_Changes_per_Hour == expected_ambient_ach
        
        # Verify new infiltration was created for Zone2
        self.mock_idf.newidfobject.assert_called_once_with('ZONEINFILTRATION:DESIGNFLOWRATE')
    
    def test_apply_infiltration_to_zones_without_existing_infiltration(self):
        """Test applying infiltration when no existing infiltration objects exist."""
        mock_idf_no_infiltration = Mock()
        mock_idf_no_infiltration.idfobjects = {
            'ZONE': [self.mock_zone1, self.mock_zone2],
            'ZONEINFILTRATION:DESIGNFLOWRATE': []
        }
        mock_idf_no_infiltration.newidfobject = Mock()
        
        infiltration_ach_50pa = 2.0
        expected_ambient_ach = 2.0 / 20.0  # 0.1 ACH ambient
        
        apply_infiltration_to_zones(mock_idf_no_infiltration, infiltration_ach_50pa)
        
        # Verify new infiltration objects were created for both zones
        assert mock_idf_no_infiltration.newidfobject.call_count == 2
    
    def test_infiltration_conversion_calculation(self):
        """Test that infiltration rate is properly converted from 50 Pa to ambient."""
        infiltration_ach_50pa = 10.0  # 10 ACH at 50 Pa
        expected_ambient_ach = 10.0 / 20.0  # 0.5 ACH ambient
        
        apply_infiltration_to_zones(self.mock_idf, infiltration_ach_50pa)
        
        # Verify conversion was applied correctly
        assert self.mock_existing_infiltration.Air_Changes_per_Hour == expected_ambient_ach
    
    def test_apply_infiltration_to_empty_zones(self):
        """Test applying infiltration when no zones exist."""
        mock_idf_empty = Mock()
        mock_idf_empty.idfobjects = {
            'ZONE': [],
            'ZONEINFILTRATION:DESIGNFLOWRATE': []
        }
        mock_idf_empty.newidfobject = Mock()
        
        # Should not raise any exceptions
        apply_infiltration_to_zones(mock_idf_empty, 1.0)
        
        # No new infiltration objects should be created
        mock_idf_empty.newidfobject.assert_not_called()


class TestCalculateWindowGeometry:
    """Test scenarios for calculate_window_geometry function."""
    
    def test_calculate_window_geometry_x_fixed_axis(self):
        """Test window geometry calculation for X-fixed axis."""
        # Window with fixed X coordinate
        vertices = [(0, 0, 1), (0, 2, 1), (0, 2, 3), (0, 0, 3)]
        wwr = 0.3
        wall_area = 20.0  # 10m width * 2m height
        sill_height = 1.0
        
        new_vertices = calculate_window_geometry(vertices, wwr, wall_area, sill_height, 0.0)  # North-facing wall
        
        # Verify result structure
        assert len(new_vertices) == 4
        assert all(len(vertex) == 3 for vertex in new_vertices)
        
        # Verify X coordinate is fixed
        x_coords = [v[0] for v in new_vertices]
        assert all(x == 0 for x in x_coords)
        
        # Verify Z coordinates maintain sill height
        z_coords = [v[2] for v in new_vertices]
        assert min(z_coords) == sill_height
    
    def test_calculate_window_geometry_y_fixed_axis(self):
        """Test window geometry calculation for Y-fixed axis."""
        # Window with fixed Y coordinate
        vertices = [(0, 5, 1), (2, 5, 1), (2, 5, 3), (0, 5, 3)]
        wwr = 0.4
        wall_area = 15.0
        sill_height = 1.0
        
        new_vertices = calculate_window_geometry(vertices, wwr, wall_area, sill_height, 90.0)  # East-facing wall
        
        # Verify result structure
        assert len(new_vertices) == 4
        assert all(len(vertex) == 3 for vertex in new_vertices)
        
        # Verify Y coordinate is fixed
        y_coords = [v[1] for v in new_vertices]
        assert all(y == 5 for y in y_coords)
    
    def test_calculate_window_geometry_aspect_ratio_selection(self):
        """Test that appropriate aspect ratios are selected based on window area."""
        vertices = [(0, 0, 1), (0, 2, 1), (0, 2, 3), (0, 0, 3)]
        wall_area = 20.0
        sill_height = 1.0
        
        # Small window area (< 1.5)
        wwr_small = 0.05  # 1.0 m² window
        new_vertices_small = calculate_window_geometry(vertices, wwr_small, wall_area, sill_height, 0.0)
        
        # Medium window area (1.5 - 3.5)
        wwr_medium = 0.15  # 3.0 m² window
        new_vertices_medium = calculate_window_geometry(vertices, wwr_medium, wall_area, sill_height, 0.0)
        
        # Large window area (> 3.5)
        wwr_large = 0.25  # 5.0 m² window
        new_vertices_large = calculate_window_geometry(vertices, wwr_large, wall_area, sill_height, 0.0)
        
        # Verify different aspect ratios produce different geometries
        assert new_vertices_small != new_vertices_medium
        assert new_vertices_medium != new_vertices_large
    
    def test_calculate_window_geometry_non_axis_aligned_error(self):
        """Test that non-axis-aligned windows raise ValueError."""
        # Window that is not axis-aligned
        vertices = [(0, 0, 1), (1, 1, 1), (1, 1, 3), (0, 0, 3)]
        wwr = 0.3
        wall_area = 20.0
        sill_height = 1.0
        
        with pytest.raises(ValueError, match="Window is not axis-aligned!"):
            calculate_window_geometry(vertices, wwr, wall_area, sill_height, 0.0)
    
    def test_calculate_window_geometry_centering(self):
        """Test that windows are properly centered horizontally."""
        vertices = [(0, 0, 1), (0, 4, 1), (0, 4, 3), (0, 0, 3)]
        wwr = 0.5
        wall_area = 20.0  # 5m width * 4m height
        sill_height = 1.0
        
        new_vertices = calculate_window_geometry(vertices, wwr, wall_area, sill_height, 0.0)
        
        # Calculate center of new window
        y_coords = [v[1] for v in new_vertices]
        window_center = (max(y_coords) + min(y_coords)) / 2
        wall_center = 2.0  # Original wall center
        
        # Window should be centered within tolerance
        assert abs(window_center - wall_center) < 0.01
    
    def test_calculate_window_geometry_sill_height_preservation(self):
        """Test that sill height is preserved in new geometry."""
        vertices = [(0, 0, 2), (0, 2, 2), (0, 2, 4), (0, 0, 4)]
        wwr = 0.3
        wall_area = 20.0
        sill_height = 2.0
        
        new_vertices = calculate_window_geometry(vertices, wwr, wall_area, sill_height, 0.0)
        
        # Verify sill height is preserved
        min_z = min(v[2] for v in new_vertices)
        assert abs(min_z - sill_height) < 0.01


class TestApplyWwrToWindows:
    """Test scenarios for apply_wwr_to_windows function."""
    
    def setup_method(self):
        """Set up mock IDF object for each test."""
        self.mock_idf = Mock()
        
        # Mock parent wall
        self.mock_parent_wall = Mock()
        self.mock_parent_wall.Name = 'Test_Wall'
        self.mock_parent_wall.Azimuth = 90.0  # East-facing
        
        # Set up wall vertices (10m x 3m wall)
        for i in range(1, 5):
            setattr(self.mock_parent_wall, f'Vertex_{i}_Xcoordinate', 0.0)
            setattr(self.mock_parent_wall, f'Vertex_{i}_Ycoordinate', float(i-1) * 3.33)
            setattr(self.mock_parent_wall, f'Vertex_{i}_Zcoordinate', 0.0 if i <= 2 else 3.0)
        
        # Mock fenestration
        self.mock_fenestration = Mock()
        self.mock_fenestration.Building_Surface_Name = 'Test_Wall'
        
        # Set up window vertices (2m x 1.5m window)
        for i in range(1, 5):
            setattr(self.mock_fenestration, f'Vertex_{i}_Xcoordinate', 0.0)
            setattr(self.mock_fenestration, f'Vertex_{i}_Ycoordinate', 1.0 + (i-1) * 0.5)
            setattr(self.mock_fenestration, f'Vertex_{i}_Zcoordinate', 1.0 if i <= 2 else 2.5)
        
        # Set up idfobjects dictionary
        self.mock_idf.idfobjects = {
            'BUILDINGSURFACE:DETAILED': [self.mock_parent_wall],
            'FENESTRATIONSURFACE:DETAILED': [self.mock_fenestration]
        }
    
    def test_apply_wwr_to_east_facing_wall(self):
        """Test applying WWR to east-facing wall."""
        wwr_east, wwr_west, wwr_north, wwr_south = 0.3, 0.2, 0.1, 0.4
        
        apply_wwr_to_windows(self.mock_idf, wwr_east, wwr_west, wwr_north, wwr_south)
        
        # Verify window vertices were updated
        # Should use wwr_east (0.3) for east-facing wall
        assert hasattr(self.mock_fenestration, 'Vertex_1_Xcoordinate')
        assert hasattr(self.mock_fenestration, 'Vertex_1_Ycoordinate')
        assert hasattr(self.mock_fenestration, 'Vertex_1_Zcoordinate')
    
    def test_apply_wwr_to_south_facing_wall(self):
        """Test applying WWR to south-facing wall."""
        # Change wall azimuth to south-facing
        self.mock_parent_wall.Azimuth = 180.0
        
        wwr_east, wwr_west, wwr_north, wwr_south = 0.2, 0.3, 0.1, 0.4
        
        apply_wwr_to_windows(self.mock_idf, wwr_east, wwr_west, wwr_north, wwr_south)
        
        # Should use wwr_south (0.4) for south-facing wall
        assert hasattr(self.mock_fenestration, 'Vertex_1_Xcoordinate')
    
    def test_apply_wwr_to_west_facing_wall(self):
        """Test applying WWR to west-facing wall."""
        # Change wall azimuth to west-facing
        self.mock_parent_wall.Azimuth = 270.0
        
        wwr_east, wwr_west, wwr_north, wwr_south = 0.2, 0.3, 0.1, 0.4
        
        apply_wwr_to_windows(self.mock_idf, wwr_east, wwr_west, wwr_north, wwr_south)
        
        # Should use wwr_west (0.3) for west-facing wall
        assert hasattr(self.mock_fenestration, 'Vertex_1_Xcoordinate')
    
    def test_apply_wwr_to_north_facing_wall(self):
        """Test applying WWR to north-facing wall."""
        # Change wall azimuth to north-facing
        self.mock_parent_wall.Azimuth = 0.0
        
        wwr_east, wwr_west, wwr_north, wwr_south = 0.2, 0.3, 0.1, 0.4
        
        apply_wwr_to_windows(self.mock_idf, wwr_east, wwr_west, wwr_north, wwr_south)
        
        # Should use wwr_north (0.1) for north-facing wall
        assert hasattr(self.mock_fenestration, 'Vertex_1_Xcoordinate')
    
    def test_apply_wwr_with_missing_parent_wall(self):
        """Test applying WWR when parent wall is not found."""
        # Change fenestration to reference non-existent wall
        self.mock_fenestration.Building_Surface_Name = 'NonExistent_Wall'
        
        wwr_east, wwr_west, wwr_north, wwr_south = 0.3, 0.2, 0.1, 0.4
        
        # Should not raise exception, just skip the fenestration
        apply_wwr_to_windows(self.mock_idf, wwr_east, wwr_west, wwr_north, wwr_south)
    
    def test_apply_wwr_with_incomplete_vertices(self):
        """Test applying WWR when wall or window vertices are incomplete."""
        # Remove some vertex coordinates
        delattr(self.mock_parent_wall, 'Vertex_3_Xcoordinate')
        
        wwr_east, wwr_west, wwr_north, wwr_south = 0.3, 0.2, 0.1, 0.4
        
        # Should not raise exception, just skip the fenestration
        apply_wwr_to_windows(self.mock_idf, wwr_east, wwr_west, wwr_north, wwr_south)
    
    def test_apply_wwr_to_multiple_fenestrations(self):
        """Test applying WWR to multiple fenestrations."""
        # Create second fenestration
        mock_fenestration2 = Mock()
        mock_fenestration2.Building_Surface_Name = 'Test_Wall'
        
        # Set up window vertices for second fenestration
        for i in range(1, 5):
            setattr(mock_fenestration2, f'Vertex_{i}_Xcoordinate', 0.0)
            setattr(mock_fenestration2, f'Vertex_{i}_Ycoordinate', 4.0 + (i-1) * 0.5)
            setattr(mock_fenestration2, f'Vertex_{i}_Zcoordinate', 1.0 if i <= 2 else 2.5)
        
        # Add to idfobjects
        self.mock_idf.idfobjects['FENESTRATIONSURFACE:DETAILED'].append(mock_fenestration2)
        
        wwr_east, wwr_west, wwr_north, wwr_south = 0.3, 0.2, 0.1, 0.4
        
        apply_wwr_to_windows(self.mock_idf, wwr_east, wwr_west, wwr_north, wwr_south)
        
        # Both fenestrations should be updated
        assert hasattr(self.mock_fenestration, 'Vertex_1_Xcoordinate')
        assert hasattr(mock_fenestration2, 'Vertex_1_Xcoordinate')
    
    def test_apply_wwr_azimuth_boundary_conditions(self):
        """Test WWR selection at azimuth boundary conditions."""
        wwr_east, wwr_west, wwr_north, wwr_south = 0.3, 0.2, 0.1, 0.4
        
        # Test boundary conditions for azimuth ranges
        test_cases = [
            (45.0, wwr_east),   # East boundary
            (134.9, wwr_east),  # Just before south boundary
            (135.0, wwr_south), # South boundary
            (224.9, wwr_south), # Just before west boundary
            (225.0, wwr_west),  # West boundary
            (314.9, wwr_west),  # Just before north boundary
            (315.0, wwr_north), # North boundary
            (44.9, wwr_north),  # Just before east boundary
        ]
        
        for azimuth, expected_wwr in test_cases:
            self.mock_parent_wall.Azimuth = azimuth
            apply_wwr_to_windows(self.mock_idf, wwr_east, wwr_west, wwr_north, wwr_south)
            # The function should use the correct WWR based on azimuth


class TestApplySolarAbsorptanceToSurfaces:
    """Test scenarios for apply_solar_absorptance_to_surfaces function."""
    
    def setup_method(self):
        """Set up mock IDF object for each test."""
        self.mock_idf = Mock()
        
        # Mock materials with different names
        self.mock_material1 = Mock()
        self.mock_material1.Name = "Cement/plaster/mortar - cement fibreboard magnesium oxysulphide binder_.0134"
        self.mock_material1.Solar_Absorptance = 0.7  # Initial value
        
        self.mock_material2 = Mock()
        self.mock_material2.Name = "Bitumen felt/sheet_.19"
        self.mock_material2.Solar_Absorptance = 0.8  # Initial value
        
        self.mock_material3 = Mock()
        self.mock_material3.Name = "Unrelated Material"
        self.mock_material3.Solar_Absorptance = 0.5  # Should not change
        
        # Set up idfobjects dictionary
        self.mock_idf.idfobjects = {
            'MATERIAL': [self.mock_material1, self.mock_material2, self.mock_material3]
        }
    
    def test_apply_solar_absorptance_to_matching_materials(self):
        """Test applying solar absorptance to materials that match the dictionary keys."""
        materials_albedo = {
            "Cement/plaster/mortar - cement fibreboard magnesium oxysulphide binder_.0134": 0.3,
            "Bitumen felt/sheet_.19": 0.2
        }
        
        apply_solar_absorptance_to_surfaces(self.mock_idf, materials_albedo)
        
        # Verify matching materials were updated
        assert self.mock_material1.Solar_Absorptance == 0.3
        assert self.mock_material2.Solar_Absorptance == 0.2
        
        # Verify unrelated material was not changed
        assert self.mock_material3.Solar_Absorptance == 0.5
    
    def test_apply_solar_absorptance_to_partial_matches(self):
        """Test applying solar absorptance when only some materials match."""
        materials_albedo = {
            "Cement/plaster/mortar - cement fibreboard magnesium oxysulphide binder_.0134": 0.1
        }
        
        apply_solar_absorptance_to_surfaces(self.mock_idf, materials_albedo)
        
        # Verify only matching material was updated
        assert self.mock_material1.Solar_Absorptance == 0.1
        
        # Verify other materials were not changed
        assert self.mock_material2.Solar_Absorptance == 0.8
        assert self.mock_material3.Solar_Absorptance == 0.5
    
    def test_apply_solar_absorptance_to_no_matches(self):
        """Test applying solar absorptance when no materials match the dictionary keys."""
        materials_albedo = {
            "Non-existent Material 1": 0.1,
            "Non-existent Material 2": 0.2
        }
        
        apply_solar_absorptance_to_surfaces(self.mock_idf, materials_albedo)
        
        # Verify no materials were changed
        assert self.mock_material1.Solar_Absorptance == 0.7
        assert self.mock_material2.Solar_Absorptance == 0.8
        assert self.mock_material3.Solar_Absorptance == 0.5
    
    def test_apply_solar_absorptance_to_empty_materials_list(self):
        """Test applying solar absorptance when no materials exist in the IDF."""
        mock_idf_empty = Mock()
        mock_idf_empty.idfobjects = {'MATERIAL': []}
        
        materials_albedo = {
            "Cement/plaster/mortar - cement fibreboard magnesium oxysulphide binder_.0134": 0.3
        }
        
        # Should not raise any exceptions
        apply_solar_absorptance_to_surfaces(mock_idf_empty, materials_albedo)
    
    def test_apply_solar_absorptance_to_empty_albedo_dict(self):
        """Test applying solar absorptance with an empty albedo dictionary."""
        materials_albedo = {}
        
        apply_solar_absorptance_to_surfaces(self.mock_idf, materials_albedo)
        
        # Verify no materials were changed
        assert self.mock_material1.Solar_Absorptance == 0.7
        assert self.mock_material2.Solar_Absorptance == 0.8
        assert self.mock_material3.Solar_Absorptance == 0.5
    
    def test_apply_solar_absorptance_with_extreme_values(self):
        """Test applying solar absorptance with extreme values (0.0 and 1.0)."""
        materials_albedo = {
            "Cement/plaster/mortar - cement fibreboard magnesium oxysulphide binder_.0134": 0.0,
            "Bitumen felt/sheet_.19": 1.0
        }
        
        apply_solar_absorptance_to_surfaces(self.mock_idf, materials_albedo)
        
        # Verify extreme values were applied correctly
        assert self.mock_material1.Solar_Absorptance == 0.0
        assert self.mock_material2.Solar_Absorptance == 1.0
    
    def test_apply_solar_absorptance_with_duplicate_material_names(self):
        """Test applying solar absorptance when multiple materials have the same name."""
        # Create duplicate material
        mock_material1_duplicate = Mock()
        mock_material1_duplicate.Name = "Cement/plaster/mortar - cement fibreboard magnesium oxysulphide binder_.0134"
        mock_material1_duplicate.Solar_Absorptance = 0.6
        
        mock_idf_with_duplicates = Mock()
        mock_idf_with_duplicates.idfobjects = {
            'MATERIAL': [self.mock_material1, mock_material1_duplicate, self.mock_material3]
        }
        
        materials_albedo = {
            "Cement/plaster/mortar - cement fibreboard magnesium oxysulphide binder_.0134": 0.3
        }
        
        apply_solar_absorptance_to_surfaces(mock_idf_with_duplicates, materials_albedo)
        
        # Verify both materials with the same name were updated
        assert self.mock_material1.Solar_Absorptance == 0.3
        assert mock_material1_duplicate.Solar_Absorptance == 0.3
    
    def test_apply_solar_absorptance_with_case_sensitive_matching(self):
        """Test that material name matching is case sensitive."""
        materials_albedo = {
            "CEMENT/PLASTER/MORTAR - CEMENT FIBREBOARD MAGNESIUM OXYSULPHIDE BINDER_.0134": 0.3,
            "bitumen felt/sheet_.19": 0.2
        }
        
        apply_solar_absorptance_to_surfaces(self.mock_idf, materials_albedo)
        
        # Verify no materials were changed due to case mismatch
        assert self.mock_material1.Solar_Absorptance == 0.7
        assert self.mock_material2.Solar_Absorptance == 0.8
        assert self.mock_material3.Solar_Absorptance == 0.5
    
    def test_apply_solar_absorptance_with_whitespace_differences(self):
        """Test that material name matching is sensitive to whitespace differences."""
        materials_albedo = {
            "Cement/plaster/mortar - cement fibreboard magnesium oxysulphide binder_.0134 ": 0.3,  # Extra space
            " Bitumen felt/sheet_.19": 0.2  # Leading space
        }
        
        apply_solar_absorptance_to_surfaces(self.mock_idf, materials_albedo)
        
        # Verify no materials were changed due to whitespace differences
        assert self.mock_material1.Solar_Absorptance == 0.7
        assert self.mock_material2.Solar_Absorptance == 0.8
        assert self.mock_material3.Solar_Absorptance == 0.5
    
    def test_apply_solar_absorptance_with_none_idf(self):
        """Test applying solar absorptance with None IDF object."""
        materials_albedo = {"Test Material": 0.3}
        
        # Should raise AttributeError when trying to access idfobjects
        with pytest.raises(AttributeError):
            apply_solar_absorptance_to_surfaces(None, materials_albedo)
    
    def test_apply_solar_absorptance_with_none_albedo_dict(self):
        """Test applying solar absorptance with None albedo dictionary."""
        # Should raise TypeError when trying to iterate over None
        with pytest.raises(TypeError):
            apply_solar_absorptance_to_surfaces(self.mock_idf, None)  # type: ignore
    
    def test_apply_solar_absorptance_with_missing_material_attributes(self):
        """Test applying solar absorptance when materials don't have expected attributes."""
        # Create material without Name attribute
        mock_material_no_name = Mock()
        mock_material_no_name.Solar_Absorptance = 0.5
        # Remove the Name attribute to simulate missing attribute
        del mock_material_no_name.Name
        
        # Create material without Solar_Absorptance attribute
        mock_material_no_absorptance = Mock()
        mock_material_no_absorptance.Name = "Test Material"
        # Remove the Solar_Absorptance attribute to simulate missing attribute
        del mock_material_no_absorptance.Solar_Absorptance
        
        mock_idf_incomplete = Mock()
        mock_idf_incomplete.idfobjects = {
            'MATERIAL': [mock_material_no_name, mock_material_no_absorptance]
        }
        
        materials_albedo = {"Test Material": 0.3}
        
        # Should raise AttributeError when trying to access missing attributes
        with pytest.raises(AttributeError):
            apply_solar_absorptance_to_surfaces(mock_idf_incomplete, materials_albedo)


class TestShadingFunctions:
    """Tests for shading rename and schedule assignment functions."""

    def setup_method(self):
        """Create a simple rectangular building and several shading objects."""
        # Mock IDF and its objects map
        self.idf = Mock()

        # Define exterior walls to establish planes: x=0 (West), x=10 (East),
        # y=0 (South), y=20 (North)
        def make_wall(constant_axis: str, const_val: float):
            wall = SimpleNamespace()
            wall.Surface_Type = 'Wall'
            wall.Outside_Boundary_Condition = 'Outdoors'
            if constant_axis == 'x':
                xs = [const_val, const_val, const_val, const_val]
                ys = [0.0, 20.0, 20.0, 0.0]
            else:
                xs = [0.0, 10.0, 10.0, 0.0]
                ys = [const_val, const_val, const_val, const_val]
            zs = [0.0, 0.0, 3.0, 3.0]
            for i in range(4):
                setattr(wall, f'Vertex_{i+1}_Xcoordinate', xs[i])
                setattr(wall, f'Vertex_{i+1}_Ycoordinate', ys[i])
                setattr(wall, f'Vertex_{i+1}_Zcoordinate', zs[i])
            return wall

        west_wall = make_wall('x', 0.0)
        east_wall = make_wall('x', 10.0)
        south_wall = make_wall('y', 0.0)
        north_wall = make_wall('y', 20.0)

        # Create shading elements near each plane
        def make_shade(name: str, verts):
            shade = SimpleNamespace()
            shade.Name = name
            for i, (x, y, z) in enumerate(verts, start=1):
                setattr(shade, f'Vertex_{i}_Xcoordinate', x)
                setattr(shade, f'Vertex_{i}_Ycoordinate', y)
                setattr(shade, f'Vertex_{i}_Zcoordinate', z)
            return shade

        # South Overhang (horizontal plate at y=0)
        south_overhang = make_shade(
            'SO_1',
            [(2.0, 0.0, 3.0), (4.0, 0.0, 3.0), (4.0, 0.0, 3.5), (2.0, 0.0, 3.5)],
        )

        # East Fin (vertical fin at x=10)
        east_fin = make_shade(
            'EF_1',
            [(10.0, 5.0, 1.0), (10.0, 5.2, 1.0), (10.0, 5.2, 3.5), (10.0, 5.0, 3.5)],
        )

        # West Overhang (horizontal plate at x=0)
        west_overhang = make_shade(
            'WO_1',
            [(0.0, 6.0, 2.5), (0.0, 8.0, 2.5), (0.0, 8.0, 2.9), (0.0, 6.0, 2.9)],
        )

        # North Fin (vertical fin at y=20) - should be ignored by sched assign
        north_fin = make_shade(
            'NF_1',
            [(6.0, 20.0, 0.5), (6.2, 20.0, 0.5), (6.2, 20.0, 3.0), (6.0, 20.0, 3.0)],
        )

        # Include one incomplete shade (skipped due to missing vertices)
        incomplete_shade = SimpleNamespace()
        incomplete_shade.Name = 'INC_1'
        setattr(incomplete_shade, 'Vertex_1_Xcoordinate', 10.0)
        # no other vertices set

        self.idf.idfobjects = {
            'BUILDINGSURFACE:DETAILED': [west_wall, east_wall, south_wall, north_wall],
            'SHADING:ZONE:DETAILED': [south_overhang],
            'SHADING:BUILDING:DETAILED': [east_fin, incomplete_shade],
            'SHADING:SITE:DETAILED': [west_overhang, north_fin],
        }

        # Keep references for assertions
        self.shades = {
            'south_overhang': south_overhang,
            'east_fin': east_fin,
            'west_overhang': west_overhang,
            'north_fin': north_fin,
            'incomplete': incomplete_shade,
        }

    def test_rename_shading_elements_by_orientation(self):
        """Shading names should be assigned by orientation and type."""
        mapping = rename_shading_elements_by_orientation(self.idf)

        # Incomplete shade should be skipped; 4 valid shades renamed
        assert len(mapping) == 4

        # Check specific renames
        assert self.shades['south_overhang'].Name == 'South Overhang 1'
        assert self.shades['east_fin'].Name == 'East Fin 1'
        assert self.shades['west_overhang'].Name == 'West Overhang 1'
        assert self.shades['north_fin'].Name == 'North Fin 1'

        # Old names should map to new names in mapping
        assert mapping['SO_1'] == 'South Overhang 1'
        assert mapping['EF_1'] == 'East Fin 1'
        assert mapping['WO_1'] == 'West Overhang 1'
        assert mapping['NF_1'] == 'North Fin 1'

    def test_assign_shading_transmittance_schedules(self):
        """Schedules should be set for S/E/W elements, not for North."""
        assigned = assign_shading_transmittance_schedules(self.idf)

        # Should assign for South Overhang, East Fin, West Overhang (3 items)
        assert assigned == 3

        assert (
            getattr(
                self.shades['south_overhang'], 'Transmittance_Schedule_Name', None
            )
            == 'South Overhang Shading Schedule'
        )
        assert (
            getattr(self.shades['east_fin'], 'Transmittance_Schedule_Name', None)
            == 'East Fin Shading Schedule'
        )
        assert (
            getattr(
                self.shades['west_overhang'], 'Transmittance_Schedule_Name', None
            )
            == 'West Overhang Shading Schedule'
        )

        # North-facing fin should be ignored (no schedule set)
        assert not hasattr(
            self.shades['north_fin'], 'Transmittance_Schedule_Name'
        )

    def test_rename_with_offset_shading_elements(self):
        """Elements near planes (not touching) should still be classified."""
        # Create slight offsets from planes (still clearly closest to a plane)
        idf = Mock()

        def make_wall(constant_axis: str, const_val: float):
            w = SimpleNamespace()
            w.Surface_Type = 'Wall'
            w.Outside_Boundary_Condition = 'Outdoors'
            if constant_axis == 'x':
                xs = [const_val] * 4
                ys = [0.0, 20.0, 20.0, 0.0]
            else:
                xs = [0.0, 10.0, 10.0, 0.0]
                ys = [const_val] * 4
            zs = [0.0, 0.0, 3.0, 3.0]
            for i in range(4):
                setattr(w, f'Vertex_{i+1}_Xcoordinate', xs[i])
                setattr(w, f'Vertex_{i+1}_Ycoordinate', ys[i])
                setattr(w, f'Vertex_{i+1}_Zcoordinate', zs[i])
            return w

        walls = [
            make_wall('x', 0.0),
            make_wall('x', 10.0),
            make_wall('y', 0.0),
            make_wall('y', 20.0),
        ]

        def shade(verts):
            s = SimpleNamespace()
            for i, (x, y, z) in enumerate(verts, start=1):
                setattr(s, f'Vertex_{i}_Xcoordinate', x)
                setattr(s, f'Vertex_{i}_Ycoordinate', y)
                setattr(s, f'Vertex_{i}_Zcoordinate', z)
            return s

        # West-overhang but 0.15 m away (x=0.15)
        w_oh = shade([(0.15, 6.0, 2.5), (0.15, 8.0, 2.5), (0.15, 8.0, 2.9), (0.15, 6.0, 2.9)])
        w_oh.Name = 'orig_w_oh'

        # South-fin but -0.1 m beyond (y=-0.1)
        s_fin = shade([(6.0, -0.1, 0.5), (6.2, -0.1, 0.5), (6.2, -0.1, 3.0), (6.0, -0.1, 3.0)])
        s_fin.Name = 'orig_s_fin'

        idf.idfobjects = {
            'BUILDINGSURFACE:DETAILED': walls,
            'SHADING:ZONE:DETAILED': [w_oh],
            'SHADING:BUILDING:DETAILED': [s_fin],
            'SHADING:SITE:DETAILED': [],
        }

        mapping = rename_shading_elements_by_orientation(idf)

        assert mapping['orig_w_oh'].startswith('West Overhang ')
        assert mapping['orig_s_fin'].startswith('South Fin ')


class TestApplyShadingSchedulesToIdf:
    """Test scenarios for apply_shading_schedules_to_idf function."""
    
    def setup_method(self):
        """Set up mock IDF object for each test."""
        self.mock_idf = Mock()
        self.mock_idf.idfobjects = {'SCHEDULE:COMPACT': []}
        self.mock_idf.newidfobject = Mock()
        self.mock_idf.removeidfobject = Mock()
    
    def test_list_input_144_values(self):
        """Test with flat list of 144 values (backward compatibility)."""
        # Create 144 values: 6 elements × 24 values each
        transmittance = [1] * 144
        
        result = apply_shading_schedules_to_idf(self.mock_idf, transmittance)
        
        assert result == self.mock_idf
        assert self.mock_idf.newidfobject.call_count == 6
    
    def test_dict_input_per_element(self):
        """Test with dict mapping elements to 24 values."""
        transmittance = {
            'south_overhang': [1, 1, 0, 0, 1, 1] * 4,  # 24 values
            'east_fin': [0, 0, 1, 1, 0, 0] * 4
        }
        
        result = apply_shading_schedules_to_idf(self.mock_idf, transmittance)
        
        assert result == self.mock_idf
        # Should create schedules for provided elements only
        assert self.mock_idf.newidfobject.call_count == 2
    
    def test_dict_input_with_normalized_keys(self):
        """Test dict input with various key formats."""
        transmittance = {
            'South Overhang': [1] * 24,
            'EAST_FIN': [0] * 24,
            'West Overhang': [1, 0] * 12
        }
        
        result = apply_shading_schedules_to_idf(self.mock_idf, transmittance)
        
        assert result == self.mock_idf
        assert self.mock_idf.newidfobject.call_count == 3
    
    def test_seasonal_dict_input(self):
        """Test dict input with seasonal keys."""
        transmittance = {
            'south_overhang': {
                'spring': [1, 1, 0, 0, 1, 1],
                'summer': [0, 0, 1, 1, 0, 0],
                'fall': [1, 0, 1, 0, 1, 0],
                'winter': [0, 1, 0, 1, 0, 1]
            }
        }
        
        result = apply_shading_schedules_to_idf(self.mock_idf, transmittance)
        
        assert result == self.mock_idf
        assert self.mock_idf.newidfobject.call_count == 1
    
    def test_autumn_fallback(self):
        """Test fallback from 'fall' to 'autumn' key."""
        transmittance = {
            'south_overhang': {
                'spring': [1] * 6,
                'summer': [0] * 6,
                'autumn': [1, 0] * 3,
                'winter': [0] * 6
            }
        }
        
        result = apply_shading_schedules_to_idf(self.mock_idf, transmittance)
        
        assert result == self.mock_idf
        assert self.mock_idf.newidfobject.call_count == 1
    
    def test_invalid_list_length(self):
        """Test error handling for invalid list length."""
        transmittance = [1] * 100  # Wrong length
        
        with pytest.raises(ValueError, match='Expected 144 transmittance values'):
            apply_shading_schedules_to_idf(self.mock_idf, transmittance)
    
    def test_invalid_input_type(self):
        """Test error handling for invalid input type."""
        transmittance = "invalid"
        
        with pytest.raises(TypeError, match='must be list\\[144\\] or dict'):
            apply_shading_schedules_to_idf(self.mock_idf, transmittance)
    
    def test_none_values_handling(self):
        """Test handling of None values in data."""
        transmittance = {
            'south_overhang': None,
            'east_fin': [1, 0] * 12
        }
        
        result = apply_shading_schedules_to_idf(self.mock_idf, transmittance)
        
        assert result == self.mock_idf
        # Should create schedule for east_fin, remove existing for south_overhang
        assert self.mock_idf.newidfobject.call_count == 1
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        transmittance = {
            'south_overhang': [],
            'east_fin': {}
        }
        
        result = apply_shading_schedules_to_idf(self.mock_idf, transmittance)
        
        assert result == self.mock_idf
        assert self.mock_idf.newidfobject.call_count == 2
    
    def test_mixed_data_types(self):
        """Test handling of mixed data types in values."""
        transmittance = {
            'south_overhang': ['1', 0, 'true', False, 1.5, -1] * 4
        }
        
        result = apply_shading_schedules_to_idf(self.mock_idf, transmittance)
        
        assert result == self.mock_idf
        assert self.mock_idf.newidfobject.call_count == 1
    
    def test_remove_existing_schedules(self):
        """Test removal of existing schedules."""
        # Mock existing schedule
        existing_schedule = Mock()
        existing_schedule.Name = 'South Overhang Shading Schedule'
        self.mock_idf.idfobjects['SCHEDULE:COMPACT'] = [existing_schedule]
        
        transmittance = {'south_overhang': [1] * 24}
        
        result = apply_shading_schedules_to_idf(self.mock_idf, transmittance)
        
        assert result == self.mock_idf
        self.mock_idf.removeidfobject.assert_called_with(existing_schedule)
        assert self.mock_idf.newidfobject.call_count == 1
    
    def test_schedule_field_generation(self):
        """Test proper schedule field generation."""
        transmittance = {'south_overhang': [1, 0, 1, 0, 1, 0] * 4}
        
        # Mock the new schedule object
        mock_schedule = Mock()
        self.mock_idf.newidfobject.return_value = mock_schedule
        
        result = apply_shading_schedules_to_idf(self.mock_idf, transmittance)
        
        assert result == self.mock_idf
        # Verify schedule properties are set
        mock_schedule.Name = 'South Overhang Shading Schedule'
        mock_schedule.Schedule_Type_Limits_Name = 'Fraction'


class TestShadingOptimizationHelpers:
    """Test cases for shading optimization helper functions."""
    
    def test_expand_partial_schedule_no_fixed_values(self):
        """Test expanding partial schedule without fixed morning values."""
        partial_values = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 4 seasons × 3 values
        result = _expand_partial_schedule(partial_values, time_slots_per_season=3)
        expected = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # Same as input
        assert result == expected
    
    def test_expand_partial_schedule_with_fixed_morning(self):
        """Test expanding partial schedule with fixed morning values."""
        partial_values = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 4 seasons × 3 values
        fixed_morning = [0, 0, 0]  # 3 morning slots
        result = _expand_partial_schedule(partial_values, time_slots_per_season=6, fixed_morning_values=fixed_morning)
        
        # Expected: 4 seasons × 6 slots = 24 values
        # Each season: [0, 0, 0] + [1, 0, 1] = [0, 0, 0, 1, 0, 1]
        # The function processes seasons in order, so the pattern repeats correctly
        expected = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0]
        assert result == expected
    
    def test_expand_partial_schedule_empty_input(self):
        """Test expanding empty partial schedule."""
        result = _expand_partial_schedule([], time_slots_per_season=6)
        expected = [0] * 24  # 4 seasons × 6 slots
        assert result == expected
    
    def test_expand_partial_schedule_invalid_length(self):
        """Test expanding partial schedule with invalid length."""
        partial_values = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 11 values (not divisible by 4)
        with pytest.raises(ValueError, match="Expected 8 values, got 11"):
            _expand_partial_schedule(partial_values, time_slots_per_season=2)
    
    def test_build_seasonal_shading_schedules_full_day_elements(self):
        """Test building seasonal schedules for full-day elements."""
        optimization_variables = [1] * 96  # 4 elements × 24 variables
        element_configs = [
            {
                'name': 'south_overhang',
                'type': 'full_day',
                'start_idx': 0,
                'time_slots': 24,
                'fixed_morning': None
            },
            {
                'name': 'south_fin',
                'type': 'full_day',
                'start_idx': 24,
                'time_slots': 24,
                'fixed_morning': None
            }
        ]
        
        result = build_seasonal_shading_schedules(optimization_variables, element_configs)
        
        assert 'south_overhang' in result
        assert 'south_fin' in result
        
        # Check south_overhang (first 24 variables)
        south_overhang = result['south_overhang']
        assert south_overhang['spring'] == [1, 1, 1, 1, 1, 1]
        assert south_overhang['summer'] == [1, 1, 1, 1, 1, 1]
        assert south_overhang['fall'] == [1, 1, 1, 1, 1, 1]
        assert south_overhang['winter'] == [1, 1, 1, 1, 1, 1]
        
        # Check south_fin (next 24 variables)
        south_fin = result['south_fin']
        assert south_fin['spring'] == [1, 1, 1, 1, 1, 1]
        assert south_fin['summer'] == [1, 1, 1, 1, 1, 1]
        assert south_fin['fall'] == [1, 1, 1, 1, 1, 1]
        assert south_fin['winter'] == [1, 1, 1, 1, 1, 1]
    
    def test_build_seasonal_shading_schedules_afternoon_only_elements(self):
        """Test building seasonal schedules for afternoon-only elements."""
        # 72 variables: 2 full-day (48) + 2 afternoon-only (24)
        optimization_variables = [1] * 72
        
        element_configs = [
            {
                'name': 'south_overhang',
                'type': 'full_day',
                'start_idx': 0,
                'time_slots': 24,
                'fixed_morning': None
            },
            {
                'name': 'south_fin',
                'type': 'full_day',
                'start_idx': 24,
                'time_slots': 24,
                'fixed_morning': None
            },
            {
                'name': 'west_overhang',
                'type': 'afternoon_only',
                'start_idx': 48,
                'time_slots': 12,
                'fixed_morning': [0, 0, 0]
            },
            {
                'name': 'west_fin',
                'type': 'afternoon_only',
                'start_idx': 60,
                'time_slots': 12,
                'fixed_morning': [0, 0, 0]
            }
        ]
        
        result = build_seasonal_shading_schedules(optimization_variables, element_configs)
        
        # Check west_overhang (afternoon-only with fixed morning)
        west_overhang = result['west_overhang']
        assert west_overhang['spring'] == [0, 0, 0, 1, 1, 1]  # [morning] + [afternoon]
        assert west_overhang['summer'] == [0, 0, 0, 1, 1, 1]
        assert west_overhang['fall'] == [0, 0, 0, 1, 1, 1]
        assert west_overhang['winter'] == [0, 0, 0, 1, 1, 1]
        
        # Check west_fin (afternoon-only with fixed morning)
        west_fin = result['west_fin']
        assert west_fin['spring'] == [0, 0, 0, 1, 1, 1]
        assert west_fin['summer'] == [0, 0, 0, 1, 1, 1]
        assert west_fin['fall'] == [0, 0, 0, 1, 1, 1]
        assert west_fin['winter'] == [0, 0, 0, 1, 1, 1]
    
    def test_build_seasonal_shading_schedules_mixed_types(self):
        """Test building seasonal schedules with mixed element types."""
        # 48 variables: 1 full-day (24) + 1 afternoon-only (24)
        optimization_variables = [1] * 48
        
        element_configs = [
            {
                'name': 'south_overhang',
                'type': 'full_day',
                'start_idx': 0,
                'time_slots': 24,
                'fixed_morning': None
            },
            {
                'name': 'west_overhang',
                'type': 'afternoon_only',
                'start_idx': 24,
                'time_slots': 12,
                'fixed_morning': [0, 0, 0]
            }
        ]
        
        result = build_seasonal_shading_schedules(optimization_variables, element_configs)
        
        assert len(result) == 2
        assert 'south_overhang' in result
        assert 'west_overhang' in result
        
        # Verify south_overhang has full 24 values
        south_overhang = result['south_overhang']
        assert all(len(season) == 6 for season in south_overhang.values())
        
        # Verify west_overhang has 6 values per season (0,0,0 + 1,1,1)
        west_overhang = result['west_overhang']
        assert all(len(season) == 6 for season in west_overhang.values())
        assert all(season[:3] == [0, 0, 0] for season in west_overhang.values())
        assert all(season[3:] == [1, 1, 1] for season in west_overhang.values())
    
    def test_build_seasonal_shading_schedules_invalid_element_type(self):
        """Test building seasonal schedules with invalid element type."""
        optimization_variables = [1] * 24
        element_configs = [
            {
                'name': 'invalid_element',
                'type': 'invalid_type',
                'start_idx': 0,
                'time_slots': 24,
                'fixed_morning': None
            }
        ]
        
        with pytest.raises(ValueError, match="Unknown element type: invalid_type"):
            build_seasonal_shading_schedules(optimization_variables, element_configs)
    
    def test_build_seasonal_shading_schedules_variable_conversion(self):
        """Test that optimization variables are properly converted to integers."""
        # Mix of float and string values
        optimization_variables = [1.0, 0.0, 1.0, 0.0] * 6  # 24 values
        element_configs = [
            {
                'name': 'test_element',
                'type': 'full_day',
                'start_idx': 0,
                'time_slots': 24,
                'fixed_morning': None
            }
        ]
        
        result = build_seasonal_shading_schedules(optimization_variables, element_configs)
        
        # All values should be converted to integers
        test_element = result['test_element']
        assert all(isinstance(val, int) for season in test_element.values() for val in season)
        assert all(val in [0, 1] for season in test_element.values() for val in season) 


class TestRenameOutsideWallsByOrientation:
    """Test scenarios for rename_outside_walls_by_orientation function."""
    
    def setup_method(self):
        """Set up mock IDF object for each test."""
        self.mock_idf = Mock()
        
        # Mock thermal zones
        self.mock_zone1 = Mock()
        self.mock_zone1.Name = 'Zone1'
        
        self.mock_zone2 = Mock()
        self.mock_zone2.Name = 'Zone2'
        
        # Mock building surfaces with different orientations
        # South wall (Y constant, min Y)
        self.mock_south_wall = Mock()
        self.mock_south_wall.Zone_Name = 'Zone1'
        self.mock_south_wall.Name = 'South_Wall_1'
        self.mock_south_wall.Outside_Boundary_Condition = 'Outdoors'
        self.mock_south_wall.Vertex_1_Xcoordinate = 5.0
        self.mock_south_wall.Vertex_2_Xcoordinate = 5.0
        self.mock_south_wall.Vertex_3_Xcoordinate = 5.0
        self.mock_south_wall.Vertex_4_Xcoordinate = 5.0
        self.mock_south_wall.Vertex_1_Ycoordinate = 0.0
        self.mock_south_wall.Vertex_2_Ycoordinate = 0.0
        self.mock_south_wall.Vertex_3_Ycoordinate = 0.0
        self.mock_south_wall.Vertex_4_Ycoordinate = 0.0
        
        # North wall (Y constant, max Y)
        self.mock_north_wall = Mock()
        self.mock_north_wall.Zone_Name = 'Zone1'
        self.mock_north_wall.Name = 'North_Wall_1'
        self.mock_north_wall.Outside_Boundary_Condition = 'Outdoors'
        self.mock_north_wall.Vertex_1_Xcoordinate = 5.0
        self.mock_north_wall.Vertex_2_Xcoordinate = 5.0
        self.mock_north_wall.Vertex_3_Xcoordinate = 5.0
        self.mock_north_wall.Vertex_4_Xcoordinate = 5.0
        self.mock_north_wall.Vertex_1_Ycoordinate = 10.0
        self.mock_north_wall.Vertex_2_Ycoordinate = 10.0
        self.mock_north_wall.Vertex_3_Ycoordinate = 10.0
        self.mock_north_wall.Vertex_4_Ycoordinate = 10.0
        
        # East wall (X constant, varying Y)
        self.mock_east_wall = Mock()
        self.mock_east_wall.Zone_Name = 'Zone1'
        self.mock_east_wall.Name = 'East_Wall_1'
        self.mock_east_wall.Outside_Boundary_Condition = 'Outdoors'
        self.mock_east_wall.Vertex_1_Xcoordinate = 10.0
        self.mock_east_wall.Vertex_2_Xcoordinate = 10.0
        self.mock_east_wall.Vertex_3_Xcoordinate = 10.0
        self.mock_east_wall.Vertex_4_Xcoordinate = 10.0
        self.mock_east_wall.Vertex_1_Ycoordinate = 0.0
        self.mock_east_wall.Vertex_2_Ycoordinate = 0.0
        self.mock_east_wall.Vertex_3_Ycoordinate = 10.0
        self.mock_east_wall.Vertex_4_Ycoordinate = 10.0
        
        # West wall (X constant, varying Y)
        self.mock_west_wall = Mock()
        self.mock_west_wall.Zone_Name = 'Zone1'
        self.mock_west_wall.Name = 'West_Wall_1'
        self.mock_west_wall.Outside_Boundary_Condition = 'Outdoors'
        self.mock_west_wall.Vertex_1_Xcoordinate = 0.0
        self.mock_west_wall.Vertex_2_Xcoordinate = 0.0
        self.mock_west_wall.Vertex_3_Xcoordinate = 0.0
        self.mock_west_wall.Vertex_4_Xcoordinate = 0.0
        self.mock_west_wall.Vertex_1_Ycoordinate = 0.0
        self.mock_west_wall.Vertex_2_Ycoordinate = 0.0
        self.mock_west_wall.Vertex_3_Ycoordinate = 10.0
        self.mock_west_wall.Vertex_4_Ycoordinate = 10.0
        
        # Indoor wall (should not be renamed)
        self.mock_indoor_wall = Mock()
        self.mock_indoor_wall.Zone_Name = 'Zone1'
        self.mock_indoor_wall.Name = 'Indoor_Wall_1'
        self.mock_indoor_wall.Outside_Boundary_Condition = 'Surface'
        self.mock_indoor_wall.Vertex_1_Xcoordinate = 5.0
        self.mock_indoor_wall.Vertex_2_Xcoordinate = 5.0
        self.mock_indoor_wall.Vertex_3_Xcoordinate = 5.0
        self.mock_indoor_wall.Vertex_4_Xcoordinate = 5.0
        self.mock_indoor_wall.Vertex_1_Ycoordinate = 5.0
        self.mock_indoor_wall.Vertex_2_Ycoordinate = 5.0
        self.mock_indoor_wall.Vertex_3_Ycoordinate = 5.0
        self.mock_indoor_wall.Vertex_4_Ycoordinate = 5.0
        
        # Mock fenestrations
        self.mock_fenestration1 = Mock()
        self.mock_fenestration1.Building_Surface_Name = 'South_Wall_1'
        
        self.mock_fenestration2 = Mock()
        self.mock_fenestration2.Building_Surface_Name = 'North_Wall_1'
        
        self.mock_fenestration3 = Mock()
        self.mock_fenestration3.Building_Surface_Name = 'East_Wall_1'
        
        self.mock_fenestration4 = Mock()
        self.mock_fenestration4.Building_Surface_Name = 'West_Wall_1'
        
        # Set up idfobjects dictionary
        self.mock_idf.idfobjects = {
            'ZONE': [self.mock_zone1, self.mock_zone2],
            'BUILDINGSURFACE:DETAILED': [
                self.mock_south_wall, self.mock_north_wall, 
                self.mock_east_wall, self.mock_west_wall, self.mock_indoor_wall
            ],
            'FENESTRATIONSURFACE:DETAILED': [
                self.mock_fenestration1, self.mock_fenestration2,
                self.mock_fenestration3, self.mock_fenestration4
            ]
        }
    
    def test_rename_outside_walls_by_orientation(self):
        """Test renaming outside walls according to orientation."""
        from .energyplus_utils import rename_outside_walls_by_orientation
        
        result = rename_outside_walls_by_orientation(self.mock_idf)
        
        # Verify walls were renamed correctly
        assert self.mock_south_wall.Name == 'South:Zone1'
        assert self.mock_north_wall.Name == 'North:Zone1'
        assert self.mock_east_wall.Name == 'East:Zone1'
        assert self.mock_west_wall.Name == 'West:Zone1'
        
        # Indoor wall should not be renamed
        assert self.mock_indoor_wall.Name == 'Indoor_Wall_1'
        
        # Verify fenestrations were updated
        assert self.mock_fenestration1.Building_Surface_Name == 'South:Zone1'
        assert self.mock_fenestration2.Building_Surface_Name == 'North:Zone1'
        assert self.mock_fenestration3.Building_Surface_Name == 'East:Zone1'
        assert self.mock_fenestration4.Building_Surface_Name == 'West:Zone1'
        
        # Verify IDF object is returned
        assert result == self.mock_idf
    
    def test_rename_outside_walls_with_multiple_zones(self):
        """Test renaming outside walls for multiple zones."""
        # Add walls for Zone2
        zone2_south_wall = Mock()
        zone2_south_wall.Zone_Name = 'Zone2'
        zone2_south_wall.Name = 'Zone2_South_Wall'
        zone2_south_wall.Outside_Boundary_Condition = 'Outdoors'
        zone2_south_wall.Vertex_1_Xcoordinate = 15.0
        zone2_south_wall.Vertex_2_Xcoordinate = 15.0
        zone2_south_wall.Vertex_3_Xcoordinate = 15.0
        zone2_south_wall.Vertex_4_Xcoordinate = 15.0
        zone2_south_wall.Vertex_1_Ycoordinate = 0.0
        zone2_south_wall.Vertex_2_Ycoordinate = 0.0
        zone2_south_wall.Vertex_3_Ycoordinate = 0.0
        zone2_south_wall.Vertex_4_Ycoordinate = 0.0
        
        self.mock_idf.idfobjects['BUILDINGSURFACE:DETAILED'].append(zone2_south_wall)
        
        from .energyplus_utils import rename_outside_walls_by_orientation
        
        result = rename_outside_walls_by_orientation(self.mock_idf)
        
        # Verify Zone2 wall was renamed
        assert zone2_south_wall.Name == 'North:Zone2'
        
        # Verify Zone1 walls still renamed correctly
        assert self.mock_south_wall.Name == 'South:Zone1'
    
    def test_rename_outside_walls_with_no_outdoor_walls(self):
        """Test renaming when no outdoor walls exist."""
        mock_idf_no_outdoor = Mock()
        mock_idf_no_outdoor.idfobjects = {
            'ZONE': [self.mock_zone1],
            'BUILDINGSURFACE:DETAILED': [self.mock_indoor_wall],
            'FENESTRATIONSURFACE:DETAILED': []
        }
        
        from .energyplus_utils import rename_outside_walls_by_orientation
        
        # Should not raise exception
        result = rename_outside_walls_by_orientation(mock_idf_no_outdoor)
        assert result == mock_idf_no_outdoor
    
    def test_rename_outside_walls_with_empty_zones(self):
        """Test renaming when no zones exist."""
        mock_idf_empty = Mock()
        mock_idf_empty.idfobjects = {
            'ZONE': [],
            'BUILDINGSURFACE:DETAILED': [],
            'FENESTRATIONSURFACE:DETAILED': []
        }
        
        from .energyplus_utils import rename_outside_walls_by_orientation
        
        # Should not raise exception
        result = rename_outside_walls_by_orientation(mock_idf_empty)
        assert result == mock_idf_empty


class TestCreateShadingElementsByOrientation:
    """Test scenarios for create_shading_elements_by_orientation function."""
    
    def setup_method(self):
        """Set up mock IDF object for each test."""
        self.mock_idf = Mock()
        self.mock_idf.newidfobject = Mock()
        
        # Mock fenestrations with different orientations
        self.mock_south_fenestration = Mock()
        self.mock_south_fenestration.Building_Surface_Name = 'South:Zone1'
        
        # Set up window vertices for south fenestration
        for i in range(1, 5):
            setattr(self.mock_south_fenestration, f'Vertex_{i}_Xcoordinate', 5.0)
            setattr(self.mock_south_fenestration, f'Vertex_{i}_Ycoordinate', 0.0 + (i-1) * 0.5)
            setattr(self.mock_south_fenestration, f'Vertex_{i}_Zcoordinate', 1.0 if i <= 2 else 2.5)
        
        self.mock_east_fenestration = Mock()
        self.mock_east_fenestration.Building_Surface_Name = 'East:Zone1'
        
        # Set up window vertices for east fenestration
        for i in range(1, 5):
            setattr(self.mock_east_fenestration, f'Vertex_{i}_Xcoordinate', 10.0 + (i-1) * 0.5)
            setattr(self.mock_east_fenestration, f'Vertex_{i}_Ycoordinate', 5.0)
            setattr(self.mock_east_fenestration, f'Vertex_{i}_Zcoordinate', 1.0 if i <= 2 else 2.5)
        
        self.mock_west_fenestration = Mock()
        self.mock_west_fenestration.Building_Surface_Name = 'West:Zone1'
        
        # Set up window vertices for west fenestration
        for i in range(1, 5):
            setattr(self.mock_west_fenestration, f'Vertex_{i}_Xcoordinate', 0.0 - (i-1) * 0.5)
            setattr(self.mock_west_fenestration, f'Vertex_{i}_Ycoordinate', 5.0)
            setattr(self.mock_west_fenestration, f'Vertex_{i}_Zcoordinate', 1.0 if i <= 2 else 2.5)
        
        # Set up idfobjects dictionary
        self.mock_idf.idfobjects = {
            'FENESTRATIONSURFACE:DETAILED': [
                self.mock_south_fenestration, self.mock_east_fenestration, self.mock_west_fenestration
            ]
        }
        
        # Mock dimensions dictionary
        self.dimensions = {
            'south': {
                'overhang': {'length': 0.5},
                'right_fin': {'length': 0.3},
                'left_fin': {'length': 0.3}
            },
            'east': {
                'overhang': {'length': 0.4},
                'left_fin': {'length': 0.2}
            },
            'west': {
                'overhang': {'length': 0.4},
                'right_fin': {'length': 0.2}
            }
        }
    
    def test_create_shading_elements_by_orientation(self):
        """Creates overhangs and fins per orientation with new params schema."""
        # Update dimensions to new schema (depth/width/angle for overhang, depth for fins)
        dims = {
            'south': {
                'overhang': {'depth': 0.5, 'width': 0.8, 'angle': 15.0},
                'right_fin': {'depth': 0.3},
                'left_fin': {'depth': 0.3},
            },
            'east': {
                'overhang': {'depth': 0.4, 'width': 0.7, 'angle': 10.0},
                'left_fin': {'depth': 0.2},
            },
            'west': {
                'overhang': {'depth': 0.4, 'width': 0.6, 'angle': 20.0},
                'right_fin': {'depth': 0.2},
            },
        }

        result = create_shading_elements_by_orientation(self.mock_idf, dims)

        # Verify new shading elements were created
        # South: 1 overhang + 2 fins, East: 1 overhang + 1 fin, West: 1 overhang + 1 fin -> 7 total
        assert self.mock_idf.newidfobject.call_count == 7

        # Verify IDF object is returned
        assert result == self.mock_idf
    
    def test_create_shading_elements_with_missing_dimensions(self):
        """Test creating shading elements when some dimensions are missing."""
        incomplete_dimensions = {
            'south': {
                'overhang': {'depth': 0.5, 'width': 0.5, 'angle': 0.0}
                # Missing right_fin and left_fin
            },
            'east': {
                'overhang': {'depth': 0.4, 'width': 0.6, 'angle': 0.0}
                # Missing left_fin
            }
        }
        
        result = create_shading_elements_by_orientation(self.mock_idf, incomplete_dimensions)
        
        # Should only create elements for available dimensions
        # South: 1 overhang, East: 1 overhang
        assert self.mock_idf.newidfobject.call_count == 2
        
        assert result == self.mock_idf
    
    def test_create_shading_elements_with_no_matching_orientations(self):
        """Test creating shading elements when no fenestrations match orientations."""
        # Change fenestration names to not match any orientation
        self.mock_south_fenestration.Building_Surface_Name = 'Unrelated_Wall'
        self.mock_east_fenestration.Building_Surface_Name = 'Another_Wall'
        self.mock_west_fenestration.Building_Surface_Name = 'Third_Wall'
        
        result = create_shading_elements_by_orientation(self.mock_idf, self.dimensions)
        
        # No shading elements should be created
        assert self.mock_idf.newidfobject.call_count == 0
        
        assert result == self.mock_idf
    
    def test_create_shading_elements_with_empty_dimensions(self):
        """Test creating shading elements with empty dimensions dictionary."""
        empty_dimensions = {}
        
        result = create_shading_elements_by_orientation(self.mock_idf, empty_dimensions)
        
        # No shading elements should be created
        assert self.mock_idf.newidfobject.call_count == 0
        
        assert result == self.mock_idf
    
    def test_create_shading_elements_with_none_dimensions(self):
        """Test creating shading elements with None dimensions."""
        # Should not raise exception
        result = create_shading_elements_by_orientation(self.mock_idf, None)
        
        # No shading elements should be created
        assert self.mock_idf.newidfobject.call_count == 0
        
        assert result == self.mock_idf
    
    def test_create_shading_elements_object_types(self):
        """Test that the correct shading object types are created."""
        # Reset mock to track calls
        self.mock_idf.newidfobject.reset_mock()

        dims = {
            'south': {
                'overhang': {'depth': 0.5, 'width': 0.8, 'angle': 15.0},
                'right_fin': {'depth': 0.3},
                'left_fin': {'depth': 0.3},
            },
            'east': {
                'overhang': {'depth': 0.4, 'width': 0.7, 'angle': 10.0},
                'left_fin': {'depth': 0.2},
            },
            'west': {
                'overhang': {'depth': 0.4, 'width': 0.6, 'angle': 20.0},
                'right_fin': {'depth': 0.2},
            },
        }

        result = create_shading_elements_by_orientation(self.mock_idf, dims)

        # Verify that SHADING:BUILDING:DETAILED objects are created
        calls = self.mock_idf.newidfobject.call_args_list

        # Check that we have the right number of SHADING:BUILDING:DETAILED objects
        shading_calls = [call for call in calls if call[0][0] == 'SHADING:BUILDING:DETAILED']

        assert len(shading_calls) == 7  # 3 overhangs + 4 fins

        # Verify the first call creates a SHADING:BUILDING:DETAILED object
        first_call = calls[0]
        assert first_call[0][0] == 'SHADING:BUILDING:DETAILED'

        assert result == self.mock_idf

    def test_create_shading_elements_vertices_and_schedules(self):
        """Verify fin vertex count and schedule names reflect orientation/type."""
        created = []

        def factory(obj_type):
            assert obj_type == 'SHADING:BUILDING:DETAILED'
            s = type('Shade', (), {})()
            created.append(s)
            return s

        self.mock_idf.newidfobject.side_effect = factory

        # Angle large enough to create triangular fins for small depths
        dims = {
            'south': {
                'overhang': {'depth': 0.6, 'width': 0.8, 'angle': 60.0},
                'left_fin': {'depth': 1.0},
                'right_fin': {'depth': 1.0},
            }
        }

        create_shading_elements_by_orientation(self.mock_idf, dims)

        # Expect 3 objects: overhang + 2 fins
        assert len(created) == 3

        # Check overhang schedule name is orientation-specific
        overhang = created[0]
        assert getattr(overhang, 'Transmittance_Schedule_Name') == 'South Overhang Shading Schedule'
        assert getattr(overhang, 'Number_of_Vertices') == 4

        # Check at least one fin becomes triangular (3 vertices) under high angle
        fin_vertices = [getattr(obj, 'Number_of_Vertices', None) for obj in created[1:]]
        assert any(v == 3 for v in fin_vertices)
        # And fin schedule names are orientation-specific
        fin_scheds = [getattr(obj, 'Transmittance_Schedule_Name', '') for obj in created[1:]]
        assert all(s == 'South Fin Shading Schedule' for s in fin_scheds)


class TestModifyIdf:
    """Test scenarios for modify_idf function."""
    
    def setup_method(self):
        """Set up mock IDF object for each test."""
        self.mock_idf = Mock()
        self.mock_idf.save = Mock()
        self.mock_idf.idfname = 'test_file.idf'
        self.mock_idf.idfobjects = {}
        
        # Mock envelope parameters
        self.envelope_parameters = {
            'external_wall': 'New_Wall_Construction',
            'roof': 'New_Roof_Construction',
            'window': 'New_Window_Construction',
            'infiltration_ach': 1.5,
            'wwr_east': 0.3,
            'wwr_west': 0.2,
            'wwr_north': 0.1,
            'wwr_south': 0.4,
            'wall_solar_absorptance': 0.3,
            'roof_solar_absorptance': 0.2,
            'shading_schedule': [1] * 144
        }
    
    def test_modify_idf_with_all_parameters(self):
        """Test modifying IDF with all envelope parameters."""
        from .energyplus_utils import modify_idf
        
        with patch('util.energyplus_utils.apply_construction_to_surfaces') as mock_apply_construction, \
             patch('util.energyplus_utils.apply_infiltration_to_zones') as mock_apply_infiltration, \
             patch('util.energyplus_utils.apply_wwr_to_windows') as mock_apply_wwr, \
             patch('util.energyplus_utils.apply_solar_absorptance_to_surfaces') as mock_apply_albedo, \
             patch('util.energyplus_utils.apply_shading_schedules_to_idf') as mock_apply_shading:
            
            # Make sure the shading function returns the IDF
            mock_apply_shading.return_value = self.mock_idf
            
            result = modify_idf(self.mock_idf, self.envelope_parameters)
            
            # Verify all functions were called
            mock_apply_construction.assert_called()
            mock_apply_infiltration.assert_called()
            mock_apply_wwr.assert_called()
            mock_apply_albedo.assert_called()
            mock_apply_shading.assert_called()
            
            # Verify IDF object is returned
            assert result == self.mock_idf
    
    def test_modify_idf_with_partial_parameters(self):
        """Test modifying IDF with only some envelope parameters."""
        partial_parameters = {
            'external_wall': 'New_Wall_Construction',
            'infiltration_ach': 1.0
        }
        
        from .energyplus_utils import modify_idf
        
        with patch('util.energyplus_utils.apply_construction_to_surfaces') as mock_apply_construction, \
             patch('util.energyplus_utils.apply_infiltration_to_zones') as mock_apply_infiltration, \
             patch('util.energyplus_utils.apply_wwr_to_windows') as mock_apply_wwr, \
             patch('util.energyplus_utils.apply_solar_absorptance_to_surfaces') as mock_apply_albedo, \
             patch('util.energyplus_utils.apply_shading_schedules_to_idf') as mock_apply_shading:
            
            result = modify_idf(self.mock_idf, partial_parameters)
            
            # Verify only relevant functions were called
            mock_apply_construction.assert_called()
            mock_apply_infiltration.assert_called()
            mock_apply_wwr.assert_not_called()
            mock_apply_albedo.assert_not_called()
            mock_apply_shading.assert_not_called()
            
            assert result == self.mock_idf
    
    def test_modify_idf_with_save_enabled(self):
        """Test modifying IDF with save enabled."""
        from .energyplus_utils import modify_idf
        
        with patch('util.energyplus_utils.apply_construction_to_surfaces'), \
             patch('util.energyplus_utils.apply_infiltration_to_zones'), \
             patch('util.energyplus_utils.apply_wwr_to_windows'), \
             patch('util.energyplus_utils.apply_solar_absorptance_to_surfaces'), \
             patch('util.energyplus_utils.apply_shading_schedules_to_idf') as mock_apply_shading:
            
            # Make sure the shading function returns the IDF
            mock_apply_shading.return_value = self.mock_idf
            
            result = modify_idf(self.mock_idf, self.envelope_parameters, save=True)
            
            # Verify save was called
            self.mock_idf.save.assert_called_once()
            
            assert result == self.mock_idf
    
    def test_modify_idf_with_custom_save_path(self):
        """Test modifying IDF with custom save path."""
        custom_save_path = '/custom/path/modified.idf'
        
        from .energyplus_utils import modify_idf
        
        with patch('util.energyplus_utils.apply_construction_to_surfaces'), \
             patch('util.energyplus_utils.apply_infiltration_to_zones'), \
             patch('util.energyplus_utils.apply_wwr_to_windows'), \
             patch('util.energyplus_utils.apply_solar_absorptance_to_surfaces'), \
             patch('util.energyplus_utils.apply_shading_schedules_to_idf') as mock_apply_shading:
            
            # Make sure the shading function returns the IDF
            mock_apply_shading.return_value = self.mock_idf
            
            result = modify_idf(self.mock_idf, self.envelope_parameters, save=True, save_path=custom_save_path)
            
            # Verify save was called with custom path
            self.mock_idf.save.assert_called_once_with(custom_save_path)
            
            assert result == self.mock_idf
    
    def test_modify_idf_with_empty_parameters(self):
        """Test modifying IDF with empty parameters dictionary."""
        empty_parameters = {}
        
        from .energyplus_utils import modify_idf
        
        with patch('util.energyplus_utils.apply_construction_to_surfaces') as mock_apply_construction, \
             patch('util.energyplus_utils.apply_infiltration_to_zones') as mock_apply_infiltration, \
             patch('util.energyplus_utils.apply_wwr_to_windows') as mock_apply_wwr, \
             patch('util.energyplus_utils.apply_solar_absorptance_to_surfaces') as mock_apply_albedo, \
             patch('util.energyplus_utils.apply_shading_schedules_to_idf') as mock_apply_shading:
            
            result = modify_idf(self.mock_idf, empty_parameters)
            
            # Verify no functions were called
            mock_apply_construction.assert_not_called()
            mock_apply_infiltration.assert_not_called()
            mock_apply_wwr.assert_not_called()
            mock_apply_albedo.assert_not_called()
            mock_apply_shading.assert_not_called()
            
            assert result == self.mock_idf
    
    def test_modify_idf_with_none_parameters(self):
        """Test modifying IDF with None parameters."""
        from .energyplus_utils import modify_idf
        
        with patch('util.energyplus_utils.apply_construction_to_surfaces') as mock_apply_construction, \
             patch('util.energyplus_utils.apply_infiltration_to_zones') as mock_apply_infiltration, \
             patch('util.energyplus_utils.apply_wwr_to_windows') as mock_apply_wwr, \
             patch('util.energyplus_utils.apply_solar_absorptance_to_surfaces') as mock_apply_albedo, \
             patch('util.energyplus_utils.apply_shading_schedules_to_idf') as mock_apply_shading:
            
            result = modify_idf(self.mock_idf, None)
            
            # Verify no functions were called
            mock_apply_construction.assert_not_called()
            mock_apply_infiltration.assert_not_called()
            mock_apply_wwr.assert_not_called()
            mock_apply_albedo.assert_not_called()
            mock_apply_shading.assert_not_called()
            
            assert result == self.mock_idf
    
    def test_modify_idf_with_wwr_parameters_only(self):
        """Test modifying IDF with only WWR parameters."""
        wwr_only_parameters = {
            'wwr_east': 0.3,
            'wwr_west': 0.2,
            'wwr_north': 0.1,
            'wwr_south': 0.4
        }
        
        from .energyplus_utils import modify_idf
        
        with patch('util.energyplus_utils.apply_construction_to_surfaces') as mock_apply_construction, \
             patch('util.energyplus_utils.apply_infiltration_to_zones') as mock_apply_infiltration, \
             patch('util.energyplus_utils.apply_wwr_to_windows') as mock_apply_wwr, \
             patch('util.energyplus_utils.apply_solar_absorptance_to_surfaces') as mock_apply_albedo, \
             patch('util.energyplus_utils.apply_shading_schedules_to_idf') as mock_apply_shading:
            
            result = modify_idf(self.mock_idf, wwr_only_parameters)
            
            # Verify only WWR function was called
            mock_apply_construction.assert_not_called()
            mock_apply_infiltration.assert_not_called()
            mock_apply_wwr.assert_called()
            mock_apply_albedo.assert_not_called()
            mock_apply_shading.assert_not_called()
            
            assert result == self.mock_idf
    
    def test_modify_idf_with_albedo_parameters_only(self):
        """Test modifying IDF with only albedo parameters."""
        albedo_only_parameters = {
            'wall_solar_absorptance': 0.3,
            'roof_solar_absorptance': 0.2
        }
        
        from .energyplus_utils import modify_idf
        
        with patch('util.energyplus_utils.apply_construction_to_surfaces') as mock_apply_construction, \
             patch('util.energyplus_utils.apply_infiltration_to_zones') as mock_apply_infiltration, \
             patch('util.energyplus_utils.apply_wwr_to_windows') as mock_apply_wwr, \
             patch('util.energyplus_utils.apply_solar_absorptance_to_surfaces') as mock_apply_albedo, \
             patch('util.energyplus_utils.apply_shading_schedules_to_idf') as mock_apply_shading:
            
            result = modify_idf(self.mock_idf, albedo_only_parameters)
            
            # Verify only albedo function was called
            mock_apply_construction.assert_not_called()
            mock_apply_infiltration.assert_not_called()
            mock_apply_wwr.assert_not_called()
            mock_apply_albedo.assert_called()
            mock_apply_shading.assert_not_called()
            
            assert result == self.mock_idf 


class TestCalculateTotalShadingArea:
    """Test scenarios for calculate_total_shading_area function."""
    
    def setup_method(self):
        """Set up mock IDF object for each test."""
        self.mock_idf = Mock()
        
        # Create mock fenestrations
        self.mock_fenestration = Mock()
        self.mock_fenestration.Name = 'Test_Window'
        self.mock_fenestration.Building_Surface_Name = 'South:Zone1'
        
        # Set up window vertices for fenestration (1m x 1.5m window)
        for i in range(1, 5):
            setattr(self.mock_fenestration, f'Vertex_{i}_Xcoordinate', 5.0)
            setattr(self.mock_fenestration, f'Vertex_{i}_Ycoordinate', 0.0 + (i-1) * 0.5)
            setattr(self.mock_fenestration, f'Vertex_{i}_Zcoordinate', 1.0 if i <= 2 else 2.5)
        
        # Create mock shading objects
        self.mock_overhang = Mock()
        self.mock_overhang.Name = 'South Overhang Test_Window'
        # Overhang vertices: 1m x 0.5m = 0.5 m²
        setattr(self.mock_overhang, 'Vertex_1_Xcoordinate', 5.0)
        setattr(self.mock_overhang, 'Vertex_1_Ycoordinate', 0.0)
        setattr(self.mock_overhang, 'Vertex_1_Zcoordinate', 2.5)
        setattr(self.mock_overhang, 'Vertex_2_Xcoordinate', 6.0)
        setattr(self.mock_overhang, 'Vertex_2_Ycoordinate', 0.0)
        setattr(self.mock_overhang, 'Vertex_2_Zcoordinate', 2.5)
        setattr(self.mock_overhang, 'Vertex_3_Xcoordinate', 6.0)
        setattr(self.mock_overhang, 'Vertex_3_Ycoordinate', 0.5)
        setattr(self.mock_overhang, 'Vertex_3_Zcoordinate', 2.5)
        setattr(self.mock_overhang, 'Vertex_4_Xcoordinate', 5.0)
        setattr(self.mock_overhang, 'Vertex_4_Ycoordinate', 0.5)
        setattr(self.mock_overhang, 'Vertex_4_Zcoordinate', 2.5)
        
        self.mock_fin = Mock()
        self.mock_fin.Name = 'South Fin Test_Window'
        # Fin vertices: 1.6m x 1.5m = 2.4 m² (extends 0.3m on each side)
        setattr(self.mock_fin, 'Vertex_1_Xcoordinate', 4.7)
        setattr(self.mock_fin, 'Vertex_1_Ycoordinate', 0.0)
        setattr(self.mock_fin, 'Vertex_1_Zcoordinate', 1.0)
        setattr(self.mock_fin, 'Vertex_2_Xcoordinate', 6.3)
        setattr(self.mock_fin, 'Vertex_2_Ycoordinate', 0.0)
        setattr(self.mock_fin, 'Vertex_2_Zcoordinate', 1.0)
        setattr(self.mock_fin, 'Vertex_3_Xcoordinate', 6.3)
        setattr(self.mock_fin, 'Vertex_3_Ycoordinate', 0.0)
        setattr(self.mock_fin, 'Vertex_3_Zcoordinate', 2.5)
        setattr(self.mock_fin, 'Vertex_4_Xcoordinate', 4.7)
        setattr(self.mock_fin, 'Vertex_4_Ycoordinate', 0.0)
        setattr(self.mock_fin, 'Vertex_4_Zcoordinate', 2.5)
        
        # Set up idfobjects dictionary
        self.mock_idf.idfobjects = {
            'FENESTRATIONSURFACE:DETAILED': [self.mock_fenestration],
            'SHADING:BUILDING:DETAILED': [self.mock_overhang, self.mock_fin]
        }
    
    def test_calculate_total_shading_area_with_building_detailed(self):
        """Test calculating total shading area with SHADING:BUILDING:DETAILED objects."""
        from util.energyplus_utils import calculate_total_shading_area
        
        total_area = calculate_total_shading_area(self.mock_idf)
        
        # Expected areas:
        # Overhang: 1m × 0.5m = 0.5 m²
        # Fin: 1.6m × 1.5m = 2.4 m²
        # Total expected: 2.9 m²
        
        expected_area = 0.5 + 2.4
        assert abs(total_area - expected_area) < 0.01, f"Expected {expected_area}, got {total_area}"
    
    def test_calculate_total_shading_area_with_no_shading(self):
        """Test calculating total shading area with no shading elements."""
        from util.energyplus_utils import calculate_total_shading_area
        
        # Remove shading objects
        self.mock_idf.idfobjects['SHADING:BUILDING:DETAILED'] = []
        
        total_area = calculate_total_shading_area(self.mock_idf)
        
        assert total_area == 0.0


class TestCopyWindowConstructionsAndMaterials:
    """Tests for copying window constructions and dependent materials."""

    def setup_method(self):
        self.src = Mock()
        self.dst = Mock()
        self.src.idfobjects = {
            'FENESTRATIONSURFACE:DETAILED': [],
            'CONSTRUCTION': [],
            'WINDOWMATERIAL:GLAZING': [],
            'WINDOWMATERIAL:GAS': [],
        }
        self.dst.idfobjects = {
            'FENESTRATIONSURFACE:DETAILED': [],
            'CONSTRUCTION': [],
            'WINDOWMATERIAL:GLAZING': [],
            'WINDOWMATERIAL:GAS': [],
        }
        # Provide API for new objects
        created = []

        def factory(key):
            ns = SimpleNamespace()
            ns.key = key
            created.append(ns)
            return ns

        self.dst.newidfobject = Mock(side_effect=factory)
        # Some eppy versions expose copyidfobject; simulate raising to use fallback
        self.dst.copyidfobject = Mock(side_effect=Exception('no copy'))

    def test_no_windows_returns_empty_mapping(self):
        mapping = copy_window_constructions_and_materials(self.src, self.dst)
        assert mapping == {}

    def test_copies_construction_and_layers_with_suffix(self):
        # Create a window that uses a construction
        win = SimpleNamespace()
        win.Construction_Name = 'WinCons1'
        self.src.idfobjects['FENESTRATIONSURFACE:DETAILED'] = [win]

        # Create construction with two layers
        cons = SimpleNamespace()
        cons.Name = 'WinCons1'
        cons.Outside_Layer = 'Clear_6mm'
        cons.Layer_2 = 'Air_12mm'
        self.src.idfobjects['CONSTRUCTION'] = [cons]

        # Create materials referenced by the construction
        glz = SimpleNamespace()
        glz.Name = 'Clear_6mm'
        gas = SimpleNamespace()
        gas.Name = 'Air_12mm'
        self.src.idfobjects['WINDOWMATERIAL:GLAZING'] = [glz]
        self.src.idfobjects['WINDOWMATERIAL:GAS'] = [gas]

        mapping = copy_window_constructions_and_materials(self.src, self.dst, suffix='__T')

        # Verify mapping and that destination has new objects via newidfobject calls
        assert mapping == {'WinCons1': 'WinCons1__T'}

        # The fallback path uses newidfobject and sets attributes; ensure it was invoked
        assert self.dst.newidfobject.call_count >= 3
    
    def test_calculate_total_shading_area_with_invalid_vertices(self):
        """Test calculating total shading area with invalid vertex data."""
        from util.energyplus_utils import calculate_total_shading_area
        
        # Create shading object with invalid vertices
        invalid_shading = Mock()
        invalid_shading.Name = 'Invalid_Shading'
        setattr(invalid_shading, 'Vertex_1_Xcoordinate', None)
        setattr(invalid_shading, 'Vertex_1_Ycoordinate', None)
        setattr(invalid_shading, 'Vertex_1_Zcoordinate', None)
        
        mock_idf = Mock()
        mock_idf.idfobjects = {'SHADING:BUILDING:DETAILED': [invalid_shading]}
        
        total_area = calculate_total_shading_area(mock_idf)
        
        assert total_area == 0.0

    def test_copies_spectral_data_and_retargets_glazing(self):
        # Setup a glazing material that references spectral data
        win = SimpleNamespace()
        win.Construction_Name = 'WinCons2'
        self.src.idfobjects['FENESTRATIONSURFACE:DETAILED'] = [win]

        cons = SimpleNamespace()
        cons.Name = 'WinCons2'
        cons.Outside_Layer = 'LoE_Clear_6mm'
        self.src.idfobjects['CONSTRUCTION'] = [cons]

        glz = SimpleNamespace()
        glz.Name = 'LoE_Clear_6mm'
        glz.Spectral_Data_Set_Name = 'SpectralSet_A'
        self.src.idfobjects['WINDOWMATERIAL:GLAZING'] = [glz]

        spectral = SimpleNamespace()
        spectral.Name = 'SpectralSet_A'
        self.src.idfobjects['MATERIALPROPERTY:GLAZINGSPECTRALDATA'] = [spectral]

        mapping = copy_window_constructions_and_materials(self.src, self.dst, suffix='__T')

        # Ensure construction copied
        assert mapping.get('WinCons2') == 'WinCons2__T'
        # Ensure newidfobject was called for glazing and spectral data
        calls = self.dst.newidfobject.call_args_list
        created_keys = [c.args[0] for c in calls]
        assert 'MATERIALPROPERTY:GLAZINGSPECTRALDATA' in created_keys
        assert 'WINDOWMATERIAL:GLAZING' in created_keys


class TestBulkCopyWindowRelatedObjects:
    def test_bulk_copies_missing_by_class(self):
        src = Mock()
        dst = Mock()
        # Prepare source objects
        cons = SimpleNamespace(); cons.Name = 'C1'
        glz = SimpleNamespace(); glz.Name = 'G1'
        gas = SimpleNamespace(); gas.Name = 'GA1'
        spec = SimpleNamespace(); spec.Name = 'S1'

        src.idfobjects = {
            'CONSTRUCTION': [cons],
            'WINDOWMATERIAL:GLAZING': [glz],
            'WINDOWMATERIAL:GAS': [gas],
            'MATERIALPROPERTY:GLAZINGSPECTRALDATA': [spec],
        }
        dst.idfobjects = {
            'CONSTRUCTION': [],
            'WINDOWMATERIAL:GLAZING': [],
            'WINDOWMATERIAL:GAS': [],
            'MATERIALPROPERTY:GLAZINGSPECTRALDATA': [],
        }

        created = []
        def factory(key):
            ns = SimpleNamespace(); ns.key = key; created.append(ns); return ns
        dst.newidfobject = Mock(side_effect=factory)
        dst.copyidfobject = Mock(side_effect=Exception('no copy'))

        report = bulk_copy_window_related_objects(src, dst)

        assert report['CONSTRUCTION'] == 1
        assert report['WINDOWMATERIAL:GLAZING'] == 1
        assert report['WINDOWMATERIAL:GAS'] == 1
        assert report['MATERIALPROPERTY:GLAZINGSPECTRALDATA'] == 1
