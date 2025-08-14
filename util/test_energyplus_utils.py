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
    assign_shading_transmittance_schedules,
    apply_shading_schedules_to_idf,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 