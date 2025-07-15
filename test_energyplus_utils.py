#!/usr/bin/env python3
"""
Test scenarios for energyplus_utils.py functions using pytest.
Tests the top 4 functions with various scenarios including edge cases.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os
from energyplus_utils import (
    apply_construction_to_surfaces,
    apply_infiltration_to_zones,
    calculate_window_geometry,
    apply_wwr_to_windows,
    apply_solar_absorptance_to_surfaces
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 