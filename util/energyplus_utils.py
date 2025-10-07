#!/usr/bin/env python3
"""
EnergyPlus runner with district heating and overheating analysis.
Runs an IDF file with a weather file, extracts district heating values,
calculates average overheating, and organizes output files.
"""

import os
import gc
import shutil
import math
from datetime import datetime
from eppy.modeleditor import IDF
from .simulation_utils import extract_district_heating_from_htm, parse_eso_file, extract_zone_temperatures
from overheating_analysis import analyze_overheating_with_ASHRAE, analyze_overheating_with_CIBSE59
import numpy as np
from typing import Dict, Tuple, List, Optional, Any, Set

def _collect_vertices(surface, count: int = 4) -> Tuple[List[float], List[float], List[float]]:
    """Return lists of x, y, z from up to 4 vertices on a surface.

    Handles empty strings and non-numeric values gracefully by skipping
    incomplete vertices.
    """
    def _as_float(value) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == '':
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    xs, ys, zs = [], [], []
    for i in range(1, count + 1):
        x = getattr(surface, 'Vertex_%d_Xcoordinate' % i, None)
        y = getattr(surface, 'Vertex_%d_Ycoordinate' % i, None)
        z = getattr(surface, 'Vertex_%d_Zcoordinate' % i, None)
        fx, fy, fz = _as_float(x), _as_float(y), _as_float(z)
        if fx is None or fy is None or fz is None:
            continue
        xs.append(fx)
        ys.append(fy)
        zs.append(fz)
    return xs, ys, zs


def _wall_plane_extents(idf, tol: float = 1e-6) -> Dict[str, Optional[float]]:
    """Identify exterior wall planes and return extreme x/y values.

    Returns a dict with keys: 'x_min', 'x_max', 'y_min', 'y_max'.
    Only considers walls with Outside_Boundary_Condition == 'Outdoors' and
    Surface_Type == 'Wall'. Assumes a rectangular building.
    """
    x_candidates = []
    y_candidates = []
    for wall in idf.idfobjects.get('BUILDINGSURFACE:DETAILED', []):
        if getattr(wall, 'Surface_Type', '') != 'Wall':
            continue
        if getattr(wall, 'Outside_Boundary_Condition', '') != 'Outdoors':
            continue
        xs, ys, _ = _collect_vertices(wall)
        if len(xs) < 2 or len(ys) < 2:
            continue
        if max(ys) - min(ys) <= tol:
            y_candidates.append(np.mean(ys))
        if max(xs) - min(xs) <= tol:
            x_candidates.append(np.mean(xs))

    if not x_candidates and not y_candidates:
        return {'x_min': None, 'x_max': None, 'y_min': None, 'y_max': None}

    x_min = min(x_candidates) if x_candidates else None
    x_max = max(x_candidates) if x_candidates else None
    y_min = min(y_candidates) if y_candidates else None
    y_max = max(y_candidates) if y_candidates else None
    return {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}


def _classify_shading_element(
    xs: List[float],
    ys: List[float],
    zs: List[float],
    planes: Dict[str, Optional[float]],
    tol: float = 0.2,
) -> Tuple[Optional[str], Optional[str]]:
    """Classify a shading element's orientation and type.

    The classification is tolerant to small offsets. Orientation is chosen
    as the closest exterior wall plane (East/West/North/South). If the
    element lies within ``tol`` of a plane it is considered aligned with
    that plane; otherwise, the minimal distance orientation is still used.

    Args:
        xs, ys, zs: coordinate lists
        planes: dict from _wall_plane_extents
        tol: proximity tolerance (meters) to accept near-coplanar elements

    Returns:
        Tuple (orientation, element_type) where orientation in
        {'North','South','East','West'} and element_type in
        {'Overhang','Fin'}
    """
    if not xs or not ys or not zs:
        return None, None

    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = max(zs) - min(zs)

    # Determine element type based on extents
    element_type = 'Fin' if dz > max(dx, dy) else 'Overhang'

    # Determine orientation by comparing to wall planes using proximity
    x_min, x_max = planes.get('x_min'), planes.get('x_max')
    y_min, y_max = planes.get('y_min'), planes.get('y_max')

    # Compute center-based distances to each plane
    distances: List[Tuple[str, float]] = []
    cx = np.mean(xs)
    cy = np.mean(ys)
    if x_max is not None:
        distances.append(('East', abs(cx - x_max)))
    if x_min is not None:
        distances.append(('West', abs(cx - x_min)))
    if y_max is not None:
        distances.append(('North', abs(cy - y_max)))
    if y_min is not None:
        distances.append(('South', abs(cy - y_min)))

    if not distances:
        return None, element_type

    # Prefer any plane within tolerance, else choose the closest
    distances.sort(key=lambda t: t[1])
    orientation = distances[0][0]

    return orientation, element_type


def rename_shading_elements_by_orientation(idf, tol: float = 1e-3) -> Dict[str, str]:
    """Rename shading elements according to wall orientation and type.

    - Filters exterior walls (Outdoors, Wall) and infers building planes.
    - Classifies each shading element as Overhang or Fin using extents.
    - Assigns names like 'South Overhang 1', 'West Fin 2', etc.

    Returns a dict mapping old names to new names.
    """
    planes = _wall_plane_extents(idf)
    counters = {
        ('South', 'Overhang'): 0,
        ('South', 'Fin'): 0,
        ('East', 'Overhang'): 0,
        ('East', 'Fin'): 0,
        ('West', 'Overhang'): 0,
        ('West', 'Fin'): 0,
        ('North', 'Overhang'): 0,
        ('North', 'Fin'): 0,
    }
    renamed = {}

    shading_keys = [
        'SHADING:ZONE:DETAILED',
        'SHADING:BUILDING:DETAILED',
        'SHADING:SITE:DETAILED',
    ]

    for key in shading_keys:
        for shade in idf.idfobjects.get(key, []):
            xs, ys, zs = _collect_vertices(shade)
            orientation, element_type = _classify_shading_element(
                xs, ys, zs, planes, tol=tol
            )
            if orientation is None or element_type is None:
                continue
            counters[(orientation, element_type)] += 1
            idx = counters[(orientation, element_type)]
            new_name = '%s %s %d' % (orientation, element_type, idx)
            old_name = getattr(shade, 'Name', None)
            shade.Name = new_name
            renamed[old_name or new_name] = new_name

    return renamed

def _rename_outdoor_walls(zone, surfaces, fenestrations, new_names, max):
    """Helper function to rename outdoor walls based on their position.
    
    Renames walls in a zone based on their coordinate values. The wall with
    the maximum coordinate value gets the first name from new_names, while
    other walls get the second name. Also updates fenestration references.
    
    Args:
        zone: The thermal zone containing the surfaces
        surfaces: List of building surfaces to rename
        fenestrations: List of fenestrations to update references
        new_names: List of two names [first_name, second_name]
        max: Maximum coordinate value for determining the first name
        
    Note:
        This is a helper function for rename_outside_walls_by_orientation.
        Only surfaces with Outside_Boundary_Condition == 'Outdoors' are renamed.
    """
    vertex = 'Vertex_1_Ycoordinate' if new_names[0] == 'North' else 'Vertex_1_Xcoordinate'
    for surface in surfaces:
        if getattr(surface, vertex) == max and surface.Outside_Boundary_Condition == 'Outdoors':
            new_name = f'{new_names[0]}:{zone.Name}'
            for fenestration in fenestrations:
                if fenestration.Building_Surface_Name == surface.Name:
                    fenestration.Building_Surface_Name = new_name
            surface.Name = new_name
        elif surface.Outside_Boundary_Condition == 'Outdoors':
            new_name = f'{new_names[1]}:{zone.Name}'
            for fenestration in fenestrations:
                if fenestration.Building_Surface_Name == surface.Name:
                    fenestration.Building_Surface_Name = new_name
            surface.Name = new_name


def rename_outside_walls_by_orientation(idf) -> Dict[str, str]:
    """Rename outside walls according to wall orientation and type.
    
    Analyzes the geometry of exterior walls in each thermal zone and renames
    them based on their orientation (North, South, East, West). Walls with
    constant Y-coordinates are classified as North/South, while walls with
    constant X-coordinates are classified as East/West. The function also
    updates fenestration references to maintain consistency.
    
    Args:
        idf: The IDF object containing building surfaces and fenestrations
        
    Returns:
        Dict[str, str]: Dictionary mapping old wall names to new names
        
    Note:
        Only walls with Outside_Boundary_Condition == 'Outdoors' are renamed.
        Indoor walls and other surface types are left unchanged.
    """
    thermal_zones = idf.idfobjects.get('ZONE', [])
    building_surfaces = idf.idfobjects.get('BUILDINGSURFACE:DETAILED', [])
    fenestrations = idf.idfobjects.get('FENESTRATIONSURFACE:DETAILED', [])
    
    for zone in thermal_zones:
        zone_surfaces = [surface for surface in building_surfaces if surface.Zone_Name == zone.Name]
        
        south_and_north_surfaces = []
        east_and_west_surfaces = []
        max_x = 0
        max_y = 0
        
        for surface in zone_surfaces:
            x_constant = surface.Vertex_1_Xcoordinate == surface.Vertex_2_Xcoordinate == surface.Vertex_3_Xcoordinate == surface.Vertex_4_Xcoordinate
            y_constant = surface.Vertex_1_Ycoordinate == surface.Vertex_2_Ycoordinate == surface.Vertex_3_Ycoordinate == surface.Vertex_4_Ycoordinate
            
            if y_constant:
                south_and_north_surfaces.append(surface)
                max_y = max(max_y, surface.Vertex_1_Ycoordinate)
                max_y = max(max_y, surface.Vertex_2_Ycoordinate)
                max_y = max(max_y, surface.Vertex_3_Ycoordinate)
                max_y = max(max_y, surface.Vertex_4_Ycoordinate)
            elif x_constant:
                east_and_west_surfaces.append(surface)
                max_x = max(max_x, surface.Vertex_1_Xcoordinate)
                max_x = max(max_x, surface.Vertex_2_Xcoordinate)
                max_x = max(max_x, surface.Vertex_3_Xcoordinate)
                max_x = max(max_x, surface.Vertex_4_Xcoordinate)
                
        _rename_outdoor_walls(zone, south_and_north_surfaces, fenestrations, ['North', 'South'], max_y)
        _rename_outdoor_walls(zone, east_and_west_surfaces, fenestrations, ['East', 'West'], max_x)        
    return idf

def cover_fenestrations_by_orientation(idf: IDF, orientation: List[str]):
    """Cover fenestrations by shading elements by wall orientation.
    
    Args:
        idf: IDF object to mutate.
        orientation: List of orientations to cover.
    """
    if len(orientation) == 0:
        return idf
    fenestrations = idf.idfobjects.get('FENESTRATIONSURFACE:DETAILED', [])
    
    for fenestration in fenestrations:
        for ori in orientation:
            if ori not in fenestration.Building_Surface_Name.lower():
                continue
            
            # Create the shading object
            shading = idf.newidfobject('SHADING:BUILDING:DETAILED')
            shading.Name = f"{ori.capitalize()} Overhang {fenestration.Name}"
            shading.Transmittance_Schedule_Name = 'Off'
            shading.Number_of_Vertices = 4
                        
            for i in range(1, 5):
                x = fenestration[f'Vertex_{i}_Xcoordinate']
                y = fenestration[f'Vertex_{i}_Ycoordinate']
                z = fenestration[f'Vertex_{i}_Zcoordinate']
                
                setattr(shading, f'Vertex_{i}_Xcoordinate', x)
                setattr(shading, f'Vertex_{i}_Ycoordinate', y)
                setattr(shading, f'Vertex_{i}_Zcoordinate', z)
    return idf

def create_shading_elements_by_orientation(idf: IDF, dimensions: Optional[Dict[str, Any]]):
    """Create shading elements (overhangs and fins) by wall orientation.

    For each fenestration, creates SHADING:BUILDING:DETAILED objects for the
    matching wall orientation(s). Supported orientations: 'south', 'west', 'east'.
    Orientation is detected by substring match against the fenestration's
    ``Building_Surface_Name`` (e.g., 'South:ZoneA').

    Input dimensions are per-orientation dictionaries. All keys are optional;
    missing items are skipped gracefully.

    Expected structure (per orientation key):
    - 'overhang': {'depth': float, 'width': float, 'angle': float}
        - depth: projection distance from wall [m]
        - width: relative factor for horizontal coverage across the window [0..]
        - angle: tilt angle from horizontal [deg]
    - 'left_fin': {'depth': float}  # fin on left jamb when viewed from outside
    - 'right_fin': {'depth': float} # fin on right jamb when viewed from outside

    Args:
        idf: IDF object to mutate.
        dimensions: Mapping of orientation -> parameters as described above.

    Returns:
        IDF: Same object, with added shading elements when requested.

    Notes:
    - Vertices are ordered counter-clockwise as seen from outside the building.
    - Only fenestrations whose ``Building_Surface_Name`` contains an orientation
      substring will receive elements.
    - Initial transmittance schedule names are assigned; you may override them
      later via ``assign_shading_transmittance_schedules``.
    """
    fenestrations = idf.idfobjects.get('FENESTRATIONSURFACE:DETAILED', [])
    walls = idf.idfobjects.get('BUILDINGSURFACE:DETAILED', [])
    
    # Handle None dimensions gracefully
    if dimensions is None:
        return idf
    
    for fenestration in fenestrations:
        wall = None
        
        for w in walls:
            if w.Name == fenestration.Building_Surface_Name:
                wall = w
                break
            
        wall_name = fenestration.Building_Surface_Name.lower()
        
        # Get fenestration geometry
        fen_vertices = _get_fenestration_vertices(fenestration)
        if not fen_vertices:
            continue
            
        # Create shading elements
        orientations = ['south', 'west', 'east']
        
        for orientation in orientations:
    
            if orientation in wall_name and orientation in dimensions:
                if 'overhang' in dimensions[orientation]:
                    _create_overhang(
                        idf, 
                        fenestration, 
                        fen_vertices,
                        dimensions[orientation]['overhang'], 
                        orientation,
                        wall
                    )
                if 'right_fin' in dimensions[orientation]:
                    _create_fin(
                        idf, 
                        fenestration, 
                        fen_vertices,
                        0, 
                        dimensions[orientation]['right_fin']['depth'], 
                        orientation,
                        overhang_angle = dimensions[orientation].get('overhang', {}).get('angle', 0)
                    )
                if 'left_fin' in dimensions[orientation]:
                    _create_fin(
                        idf, 
                        fenestration, 
                        fen_vertices,
                        dimensions[orientation]['left_fin']['depth'],
                        0, 
                        orientation,
                        overhang_angle = dimensions[orientation].get('overhang', {}).get('angle', 0)
                    )
    
    return idf


def _get_fenestration_vertices(fenestration: Any) -> Optional[List[Tuple[float, float, float]]]:
    """Extract fenestration vertices as a list of (x, y, z) tuples.

    Returns None if fewer than 3 valid vertices are available.
    """
    vertices = []
    for i in range(1, 5):  # EnergyPlus uses 1-based indexing
        try:
            x = getattr(fenestration, f'Vertex_{i}_Xcoordinate')
            y = getattr(fenestration, f'Vertex_{i}_Ycoordinate')
            z = getattr(fenestration, f'Vertex_{i}_Zcoordinate')
            if x is not None and y is not None and z is not None:
                vertices.append((float(x), float(y), float(z)))
        except (AttributeError, ValueError):
            continue
    
    return vertices if len(vertices) >= 3 else None


def _overhang_width_diff(ori: str, span_x: float, span_y: float, rel_width: float) -> float:
    """Compute width extension based on window width (span).

    Uses the window span for the orientation's lateral axis and scales by
    `rel_width` to determine the desired overhang width across the window.

    width_diff = (window_span * rel_width) - window_span
    where window_span is along X for south and along Y for west/east.
    """
    window_span = span_x if ori == 'south' else span_y
    return window_span * rel_width - window_span


def _create_overhang(
    idf: IDF,
    fenestration: Any,
    fen_vertices: List[Tuple[float, float, float]],
    params: Dict[str, float],
    orientation: str,
    wall: Any,
) -> None:
    """Create an overhang shading element.

    Args:
        idf: IDF object.
        fenestration: Fenestration IDF object.
        fen_vertices: Fenestration vertices as (x, y, z).
        params: Dict with keys 'depth', 'width', 'angle'.
        orientation: 'south' | 'west' | 'east'.
        wall: Parent wall object for span/width reference.
    """
    if not fen_vertices or len(fen_vertices) < 3:
        return

    depth, width, angle = params['depth'], params['width'], params['angle']
    
    # Calculate overhang geometry
    # Overhang extends outward from the building (negative Z direction for most orientations)
    # and covers the fenestration area
    
    # Find bounding box of fenestration
    xs = [v[0] for v in fen_vertices]
    ys = [v[1] for v in fen_vertices]
    zs = [v[2] for v in fen_vertices]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    _, max_z = min(zs), max(zs)
    min_z = max_z - math.tan(angle * math.pi / 180) * depth

    # Compute overhang lateral width once based on window span
    width_diff = _overhang_width_diff(orientation, max_x - min_x, max_y - min_y, width)

    # Create overhang vertices (counter-clockwise when viewed from outside)
    # Extend outward from the building surface
    if orientation == 'south':
        # South overhang extends north (negative Y)
        vertices = [
            (min_x - width_diff/2, max_y - depth, min_z),     # Vertex 1: bottom-left
            (max_x + width_diff/2, max_y - depth, min_z),     # Vertex 2: bottom-right
            (max_x + width_diff/2, max_y, max_z),             # Vertex 3: top-right
            (min_x - width_diff/2, max_y, max_z),             # Vertex 4: top-left
        ]
    elif orientation == 'west':
        # West overhang extends west (negative X)
        vertices = [
            (min_x, min_y - width_diff/2, min_z),             # Vertex 1: bottom-left
            (min_x, max_y + width_diff/2, min_z),             # Vertex 2: bottom-right
            (min_x - depth, max_y + width_diff/2, max_z),     # Vertex 3: top-right
            (min_x - depth, min_y - width_diff/2, max_z),     # Vertex 4: top-left
        ]
    elif orientation == 'east':
        # East overhang extends east (positive X)
        vertices = [
            (max_x + depth, min_y - width_diff/2, min_z),     # Vertex 1: bottom-left
            (max_x + depth, max_y + width_diff/2, min_z),     # Vertex 2: bottom-right
            (max_x, max_y + width_diff/2, max_z),             # Vertex 3: top-right
            (max_x, min_y - width_diff/2, max_z),             # Vertex 4: top-left
        ]
    else:
        return
    
    # Create the shading object
    shading = idf.newidfobject('SHADING:BUILDING:DETAILED')
    shading.Name = f"{orientation.capitalize()} Overhang {fenestration.Name}"
    shading.Transmittance_Schedule_Name = "South Overhang Shading Schedule"
    shading.Number_of_Vertices = 4
    
    # Set vertices with proper counter-clockwise ordering
    for i, (x, y, z) in enumerate(vertices, 1):
        setattr(shading, f'Vertex_{i}_Xcoordinate', x)
        setattr(shading, f'Vertex_{i}_Ycoordinate', y)
        setattr(shading, f'Vertex_{i}_Zcoordinate', z)


def _create_fin(
    idf: IDF,
    fenestration: Any,
    fen_vertices: List[Tuple[float, float, float]],
    left_depth: float,
    right_depth: float,
    orientation: str,
    overhang_angle: float = 0,
) -> None:
    """Create a vertical fin shading element.

    Args:
        idf: IDF object.
        fenestration: Fenestration IDF object.
        fen_vertices: Fenestration vertices as (x, y, z).
        left_depth: Depth [m] for left-side fin (0 to skip).
        right_depth: Depth [m] for right-side fin (0 to skip).
        orientation: 'south' | 'west' | 'east'.
        overhang_angle: Overhang tilt angle [deg] to align fin top.
    """
    if not fen_vertices or len(fen_vertices) < 3:
        return
    # Calculate fin geometry
    # Fins extend outward from the building surface perpendicular to the wall
    
    # Find bounding box of fenestration
    xs = [v[0] for v in fen_vertices]
    ys = [v[1] for v in fen_vertices]
    zs = [v[2] for v in fen_vertices]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    
    # print(min_x, max_x)
    fin_depth = left_depth if left_depth > 0 else right_depth
    middle_z = max_z - math.tan(overhang_angle * math.pi / 180) * fin_depth
    
    has_three_vertices = middle_z < min_z
    new_depth = 1.5 * math.cos(overhang_angle * math.pi / 180) if has_three_vertices else fin_depth
    # Create fin vertices (counter-clockwise when viewed from outside)
    if orientation == 'south' and left_depth > 0:
        # South fin extends south from left of fenestration (negative Y) as seen from east side of building
        vertices = [
            (min_x, min_y - new_depth, min_z),      # Vertex 1: bottom-left
            (min_x, min_y, min_z),                  # Vertex 2: bottom-right
            (min_x, min_y, max_z),                  # Vertex 3: top-right     
        ]
        
        if not has_three_vertices:
            vertices.append((
                min_x, 
                min_y - new_depth, 
                max_z - math.tan(overhang_angle * math.pi / 180) * new_depth
            ))  # Vertex 4: top-left
    elif orientation == 'south' and right_depth > 0:
        # South fin extends south from right of fenestration (negative Y) as seen from east side of building
        vertices = [
            (max_x, min_y - new_depth, min_z),      # Vertex 1: bottom-left
            (max_x, min_y, min_z),                  # Vertex 2: bottom-right
            (max_x, min_y, max_z)                   # Vertex 3: top-right     
        ]
        
        if not has_three_vertices:
            vertices.append((
                min_x, 
                min_y - new_depth, 
                max_z - math.tan(overhang_angle * math.pi / 180) * new_depth
            )) # Vertex 4: top-left

    elif orientation == 'west':
        # West fin extends west (negative X) as seen from north side of building
        vertices = [
            (min_x, min_y, max_z),                  # Vertex 1: top-left
            (min_x, min_y , min_z),                 # Vertex 2: bottom-left
            (min_x - new_depth, min_y, min_z)       # Vertex 3: bottom-right
        ]
        
        if not has_three_vertices:
            vertices.append((
                min_x - new_depth, 
                min_y, 
                max_z - math.tan(overhang_angle * math.pi / 180) * new_depth
            )) # Vertex 4: top-right

    elif orientation == 'east':
        # East fin extends east (positive X)
        vertices = [
            (max_x + new_depth, min_y, min_z),      # Vertex 1: bottom-left
            (max_x, max_y, min_z),                  # Vertex 2: bottom-right
            (max_x, max_y, max_z)                   # Vertex 3: top-right
        ]
        
        if not has_three_vertices:
            vertices.append((
                max_x + new_depth, 
                min_y, 
                max_z - math.tan(overhang_angle * math.pi / 180) * new_depth
            )) # Vertex 4: top-right
    else:
        return
    
    # Create the shading object
    shading = idf.newidfobject('SHADING:BUILDING:DETAILED')
    side = 'Left' if left_depth > 0 else 'Right'
    shading.Name = f"{orientation.capitalize()} Fin {fenestration.Name} {side}"
    shading.Transmittance_Schedule_Name = "South Overhang Shading Schedule"
    shading.Number_of_Vertices = len(vertices)
    
    # Set vertices with proper counter-clockwise ordering
    for i, (x, y, z) in enumerate(vertices, 1):
        setattr(shading, f'Vertex_{i}_Xcoordinate', x)
        setattr(shading, f'Vertex_{i}_Ycoordinate', y)
        setattr(shading, f'Vertex_{i}_Zcoordinate', z)


def assign_shading_transmittance_schedules(idf, tol: float = 1e-3) -> int:
    """Assign transmittance schedules by orientation and element type.

    South/East/West Overhangs and Fins get their respective schedules:
    '<Orientation> <Type> Shading Schedule'. North-facing elements are
    ignored as per current requirements.
    """
    planes = _wall_plane_extents(idf)
    shading_keys = [
        'SHADING:ZONE:DETAILED',
        'SHADING:BUILDING:DETAILED',
        'SHADING:SITE:DETAILED',
    ]
    assigned = 0

    for key in shading_keys:
        for shade in idf.idfobjects.get(key, []):
            xs, ys, zs = _collect_vertices(shade)
            orientation, element_type = _classify_shading_element(
                xs, ys, zs, planes, tol=tol
            )
            if orientation in ('South', 'East', 'West') and element_type in (
                'Overhang', 'Fin'
            ):
                sched = (
                    orientation
                    + ' '
                    + element_type
                    + ' Shading Schedule'
                )
                setattr(shade, 'Transmittance_Schedule_Name', sched)
                assigned += 1

    return assigned




def _normalize_transmittance_value(value):
    """Convert value to binary (0 or 1) for transmittance."""
    try:
        return 1 if int(value) != 0 else 0
    except (ValueError, TypeError):
        return 0


def _parse_seasonal_data(data):
    """Parse seasonal data into 24-value list."""
    if data is None:
        return None
    
    if isinstance(data, (list, tuple)):
        if not data:  # Empty list/tuple
            return [0] * 24
        return [_normalize_transmittance_value(v) for v in data[:24]] + [0] * max(0, 24 - len(data))
    
    if isinstance(data, dict):
        if not data:  # Empty dict
            return [0] * 24
        seasons = ['spring', 'summer', 'fall', 'winter']
        result = []
        for season in seasons:
            season_data = data.get(season, data.get('autumn' if season == 'fall' else season, []))
            season_values = [_normalize_transmittance_value(v) for v in season_data[:6]]
            result.extend(season_values + [0] * max(0, 6 - len(season_values)))
        return result
    
    return [0] * 24


def _create_schedule_fields(values_24):
    """Create schedule fields for EnergyPlus schedule object."""
    time_periods = [
        ('24:00', '06:00'), ('06:00', '08:00'), ('08:00', '10:00'),
        ('10:00', '12:00'), ('12:00', '14:00'), ('14:00', '16:00'),
        ('16:00', '18:00'), ('18:00', '24:00')
    ]
    
    season_dates = ['21 Mar', '21 Jun', '21 Sep', '21 Dec', '31 Dec']
    season_ranges = [(0, 6), (6, 12), (12, 18), (18, 24), (0, 6)]
    
    fields = []
    for (start, end), date in zip(season_ranges, season_dates):
        fields.extend(['Through: %s' % date, 'For: AllDays'])
        for i, (_, end_time) in enumerate(time_periods):
            if i == 0:
                val = 0
            elif i <= 6:
                val = values_24[start + min(i - 1, 5)]
            else:
                val = 0
            fields.extend(['Until: %s' % end_time, str(val)])
    
    return fields


def _remove_existing_schedule(idf, schedule_name):
    """Remove existing schedule with given name."""
    existing = [s for s in idf.idfobjects['SCHEDULE:COMPACT'] 
                if s.Name == schedule_name]
    for schedule in existing:
        idf.removeidfobject(schedule)


def _create_shading_schedule(idf, element_name, values_24):
    """Create and configure a shading schedule object."""
    schedule_name = f"{element_name.replace('_', ' ').title()} Shading Schedule"
    
    _remove_existing_schedule(idf, schedule_name)
    
    schedule = idf.newidfobject('SCHEDULE:COMPACT')
    schedule.Name = schedule_name
    schedule.Schedule_Type_Limits_Name = 'Fraction'
    
    fields = _create_schedule_fields(values_24)
    for i, field_value in enumerate(fields, 1):
        setattr(schedule, f'Field_{i}', field_value)


def apply_shading_schedules_to_idf(idf, transmittance):
    """
    Apply shading schedules per element using 24 values (4 seasons × 6 slots).

    Accepts either:
    - A flat list of length 144 (6 elements × 24) – backward-compatible
    - A dict mapping shading elements to a flat list of 24 values

    Per-element list format (length 24):
    [spring(6), summer(6), fall(6), winter(6)] where each season block is
    [06-08, 08-10, 10-12, 12-14, 14-16, 16-18].

    Args:
        idf: IDF object
        transmittance: list[int] of length 144 OR dict[str, list[int]] of length 24 per element

    Returns:
        IDF: Modified IDF with shading schedules applied
    """
    shading_elements = [
        'south_overhang', 'south_fin', 'west_overhang',
        'west_fin', 'east_overhang', 'east_fin'
    ]
    
    # Normalize input to element mapping
    if isinstance(transmittance, (list, tuple)):
        if len(transmittance) != 144:
            raise ValueError('Expected 144 transmittance values for list input')
        
        elements_map = {}
        for i, element in enumerate(shading_elements):
            start = i * 24
            elements_map[element] = transmittance[start:start + 24]
    
    elif isinstance(transmittance, dict):
        elements_map = {str(k).strip().lower().replace(' ', '_'): v 
                       for k, v in transmittance.items()}
    else:
        raise TypeError('transmittance must be list[144] or dict')
    
    # Process each shading element
    for element_name in shading_elements:
        if element_name not in elements_map:
            # Remove existing schedule if element not provided
            schedule_name = f"{element_name.replace('_', ' ').title()} Shading Schedule"
            _remove_existing_schedule(idf, schedule_name)
            continue
        
        values_24 = _parse_seasonal_data(elements_map[element_name])
        if values_24 is None:
            # Remove existing schedule if data is None
            schedule_name = f"{element_name.replace('_', ' ').title()} Shading Schedule"
            _remove_existing_schedule(idf, schedule_name)
            continue
        
        _create_shading_schedule(idf, element_name, values_24)
    
    return idf


def apply_construction_to_surfaces(idf, construction_name, surface_type):
    """
    Apply a construction to all surfaces of a specific type.
    
    Args:
        idf: IDF object
        construction_name (str): Name of the construction to apply
        surface_type (str): Type of surface ('Wall', 'Roof', 'Window')
    """
    if surface_type == 'Wall':
        surfaces = idf.idfobjects['BUILDINGSURFACE:DETAILED']
        for surface in surfaces:
            if surface.Surface_Type == 'Wall' and surface.Outside_Boundary_Condition == 'Outdoors':
                surface.Construction_Name = construction_name
    elif surface_type == 'Roof':
        surfaces = idf.idfobjects['BUILDINGSURFACE:DETAILED']
        for surface in surfaces:
            if surface.Surface_Type == 'Roof':
                surface.Construction_Name = construction_name
    elif surface_type == 'Window':
        fenestrations = idf.idfobjects['FENESTRATIONSURFACE:DETAILED']
        for fenestration in fenestrations:
            fenestration.Construction_Name = construction_name


def apply_infiltration_to_zones(idf, infiltration_ach_50pa):
    """
    Apply infiltration rate to all zones.
    
    Args:
        idf: IDF object
        infiltration_ach_50pa (float): Infiltration rate in ACH at 50 Pa
    """
    # Convert ACH at 50 Pa to ambient conditions (divide by 20 for typical conversion)
    infiltration_ach_ambient = infiltration_ach_50pa / 20.0
    
    zones = idf.idfobjects['ZONE']
    
    for zone in zones:
        zone_name = zone.Name
        
        # Check if infiltration object already exists for this zone
        existing_infiltration = None
        for infiltration in idf.idfobjects['ZONEINFILTRATION:DESIGNFLOWRATE']:
            if infiltration.Zone_or_ZoneList_Name == zone_name:
                existing_infiltration = infiltration
                break
        
        if existing_infiltration:
            # Update existing infiltration - clear Design_Flow_Rate and set Air_Changes_per_Hour
            existing_infiltration.Design_Flow_Rate = ''
            existing_infiltration.Air_Changes_per_Hour = infiltration_ach_ambient
            existing_infiltration.Design_Flow_Rate_Calculation_Method = 'AirChanges/Hour'
        else:
            # Create new infiltration object
            infiltration = idf.newidfobject('ZONEINFILTRATION:DESIGNFLOWRATE')
            infiltration.Zone_or_ZoneList_Name = zone_name
            infiltration.Design_Flow_Rate = ''  # Clear this field
            infiltration.Air_Changes_per_Hour = infiltration_ach_ambient  # Set converted ACH
            infiltration.Design_Flow_Rate_Calculation_Method = 'AirChanges/Hour'


def calculate_window_geometry(vertices, wwr, wall_area, sill_height, wall_azimuth):
    """
    Recompute window vertices from WWR using a fixed 1.5 m height.

    Uses window_area = wall_area * wwr. Sets height to 1.5 m and derives width
    from area/height. Preserves sill height and centers horizontally on the
    wall's variable axis. The returned vertices follow the original winding.
    """
    vertices = np.array(vertices)
    wall_azimuth = wall_azimuth % 360
    # Axis-alignment check (restore original logic)
    x_vals = vertices[:, 0]
    y_vals = vertices[:, 1]
    if (not np.allclose(x_vals, x_vals[0])) and (not np.allclose(y_vals, y_vals[0])):
        raise ValueError("Window is not axis-aligned!")
    
    # Determine wall orientation: keep window coplanar with wall
    # East/West walls: X is constant (YZ plane) → vary Y
    # North/South walls: Y is constant (XZ plane) → vary X
    if (45 <= wall_azimuth < 135) or (225 <= wall_azimuth < 315):
        fixed_axis = 'x'
        var_vals = vertices[:, 1]
    else:
        fixed_axis = 'y'
        var_vals = vertices[:, 0]
    
    # Calculate center of the variable coordinate (for centering the window)
    center = (np.max(var_vals) + np.min(var_vals)) / 2
    
    # Calculate new window area and dimensions (fixed height = 1.5 m)
    window_area = wall_area * wwr
    fixed_height = 1.5
    new_height = fixed_height
    new_width = window_area / fixed_height
    
    # Keep original sill height
    new_z_min = sill_height
    new_z_max = sill_height + new_height
    
    # Calculate new variable coordinate range (centered)
    new_var_min = center - new_width / 2
    new_var_max = center + new_width / 2
    
    # Build rectangle candidates with desired sequence:
    # Vertex 1: bottom-right, 2: bottom-left, 3: top-left, 4: top-right
    if fixed_axis == 'x':
        x_fixed = x_vals[0]
        # Candidate A: right = new_var_max, left = new_var_min
        candA = [
            (x_fixed, new_var_max, new_z_min),  # BR
            (x_fixed, new_var_min, new_z_min),  # BL
            (x_fixed, new_var_min, new_z_max),  # TL
            (x_fixed, new_var_max, new_z_max),  # TR
        ]
        # Candidate B: right = new_var_min, left = new_var_max
        candB = [
            (x_fixed, new_var_min, new_z_min),
            (x_fixed, new_var_max, new_z_min),
            (x_fixed, new_var_max, new_z_max),
            (x_fixed, new_var_min, new_z_max),
        ]
    else:
        y_fixed = y_vals[0]
        # Candidate A: right = new_var_max, left = new_var_min
        candA = [
            (new_var_max, y_fixed, new_z_min),  # BR
            (new_var_min, y_fixed, new_z_min),  # BL
            (new_var_min, y_fixed, new_z_max),  # TL
            (new_var_max, y_fixed, new_z_max),  # TR
        ]
        # Candidate B: right = new_var_min, left = new_var_max
        candB = [
            (new_var_min, y_fixed, new_z_min),
            (new_var_max, y_fixed, new_z_min),
            (new_var_max, y_fixed, new_z_max),
            (new_var_min, y_fixed, new_z_max),
        ]

    # Use appropriate projection plane for signed area
    def signed_area_on_plane(verts, plane: str) -> float:
        if plane == 'yz':
            a = [v[1] for v in verts]
            b = [v[2] for v in verts]
        elif plane == 'xz':
            a = [v[0] for v in verts]
            b = [v[2] for v in verts]
        else:  # 'xy'
            a = [v[0] for v in verts]
            b = [v[1] for v in verts]
        return 0.5 * sum(a[i] * b[(i + 1) % 4] - a[(i + 1) % 4] * b[i] for i in range(4))

    if fixed_axis == 'x':
        plane = 'yz'
    elif fixed_axis == 'y':
        plane = 'xz'
    else:
        plane = 'xy'

    orig_area = signed_area_on_plane(vertices.tolist(), plane)
    areaA = signed_area_on_plane(candA, plane)

    # Pick candidate whose winding matches original to preserve outward normal
    chosen = candA if np.sign(areaA) == np.sign(orig_area) else candB

    return chosen


def apply_wwr_to_windows(idf, params: dict | None = None):
    fenestrations = idf.idfobjects['FENESTRATIONSURFACE:DETAILED']
    for fenestration in fenestrations:
        parent_wall_name = fenestration.Building_Surface_Name
        parent_wall_name_l = str(parent_wall_name or '').lower()
        parent_wall = None
        for wall in idf.idfobjects['BUILDINGSURFACE:DETAILED']:
            if wall.Name == parent_wall_name:
                parent_wall = wall
                break
        if parent_wall is None:
            continue
        
        # Calculate azimuth from wall vertices instead of accessing non-existent field
        wall_vertices = []
        for i in range(1, 5):
            x = getattr(parent_wall, f'Vertex_{i}_Xcoordinate', None)
            y = getattr(parent_wall, f'Vertex_{i}_Ycoordinate', None)
            z = getattr(parent_wall, f'Vertex_{i}_Zcoordinate', None)
            if x is not None and y is not None and z is not None:
                wall_vertices.append((float(x), float(y), float(z)))
        
        if len(wall_vertices) < 4:
            continue
        
        # Compute azimuth once from wall geometry for downstream use
        v0, v1 = wall_vertices[0], wall_vertices[1]
        dx = v1[0] - v0[0]
        dy = v1[1] - v0[1]
        azimuth = np.degrees(np.arctan2(dy, dx)) % 360

        # Determine orientation string for zone-specific lookup
        orientation_str = None
        if 'south' in parent_wall_name_l:
            orientation_str = 'south'
        elif 'west' in parent_wall_name_l:
            orientation_str = 'west'
        elif 'east' in parent_wall_name_l:
            orientation_str = 'east'
        elif 'north' in parent_wall_name_l:
            orientation_str = 'north'

        # Fallback to azimuth-based orientation if name not informative
        if orientation_str is None:
            if 45 <= azimuth < 135:
                orientation_str = 'east'
            elif 135 <= azimuth < 225:
                orientation_str = 'north'
            elif 225 <= azimuth < 315:
                orientation_str = 'west'
            else:
                orientation_str = 'south'

        # Zone-specific precedence using parent wall zone name
        wwr = None
        if params is not None and orientation_str is not None:
            try:
                zone_name = str(getattr(parent_wall, 'Zone_Name', '') or '')
            except Exception:
                zone_name = ''
            zn = zone_name.upper()
            zone_type = None
            if 'BR' in zn:
                zone_type = 'br'
            elif 'LR' in zn:
                zone_type = 'lr'

            if zone_type is not None:
                key = f"wwr_{orientation_str}_{zone_type}"
                z_wwr = params.get(key)
                if z_wwr is not None:
                    wwr = z_wwr

        # Orientation-level fallback from params
        if wwr is None and params is not None and orientation_str is not None:
            wwr = params.get(f'wwr_{orientation_str}')

        # Skip if no WWR provided for this orientation
        if wwr is None:
            continue
        wwr = float(wwr)
        
        # Calculate wall area (assume rectangle)
        v0, v1, v2, v3 = wall_vertices
        width = np.linalg.norm(np.array(v1)[:2] - np.array(v0)[:2])
        height = np.abs(v2[2] - v0[2])
        wall_area = width * height
        
        # Get window vertices
        win_vertices = []
        for i in range(1, 5):
            x = getattr(fenestration, f'Vertex_{i}_Xcoordinate', None)
            y = getattr(fenestration, f'Vertex_{i}_Ycoordinate', None)
            z = getattr(fenestration, f'Vertex_{i}_Zcoordinate', None)
            if x is not None and y is not None and z is not None:
                win_vertices.append((float(x), float(y), float(z)))
        if len(win_vertices) < 4:
            continue
        # Sill height
        sill_height = min([v[2] for v in win_vertices])
        # Calculate new vertices
        new_vertices = calculate_window_geometry(win_vertices, wwr, wall_area, sill_height, azimuth)
        # Update fenestration vertices
        for i, (x, y, z) in enumerate(new_vertices, 1):
            setattr(fenestration, f'Vertex_{i}_Xcoordinate', x)
            setattr(fenestration, f'Vertex_{i}_Ycoordinate', y)
            setattr(fenestration, f'Vertex_{i}_Zcoordinate', z)

def apply_solar_absorptance_to_surfaces(idf, materials_albedo: dict[str, float]):
    """
    Apply albedo to all surfaces.
    
    Args:
        idf: IDF object
        materials_albedo (dict): Dictionary containing material names and albedo values
    """
    materials = idf.idfobjects['MATERIAL']
    for material in materials:
        if material.Name in materials_albedo:
            material.Solar_Absorptance = materials_albedo[material.Name]

def modify_idf(idf, parameters, save=False, save_path=None):
    """
    Apply parameters to an IDF object and optionally save the modified file.
    
    Args:
        idf: IDF object to modify
        parameters (dict): Dictionary containing envelope parameters and optional shading dimensions:
            Envelope parameters:
            - 'external_wall': External wall construction name
            - 'roof': Roof construction name
            - 'window': Window construction name
            - 'infiltration_ach': Infiltration rate in ACH at 50 Pa
            - 'wwr_east': Window-to-wall ratio for east facade
            - 'wwr_west': Window-to-wall ratio for west facade
            - 'wwr_north': Window-to-wall ratio for north facade
            - 'wwr_south': Window-to-wall ratio for south facade
            - 'shading_schedule': 144-long list (6 elements × 4 seasons × 6 periods)
            
            Shading geometry (preferred):
            - 'shading_dimensions': nested dict with keys per orientation, e.g.
                {
                  'south': {'overhang': {'length': 0.6}, 'left_fin': {'length': 0.4}},
                  'west':  {'overhang': {'length': 0.3}, 'right_fin': {'length': 0.5}}
                }
        save (bool): Whether to save the modified IDF file
        save_path (str, optional): Path to save the modified IDF file. If None, uses original path
    
    Returns:
        IDF: The modified IDF object
    """
    # Handle None parameters gracefully
    if parameters is None:
        parameters = {}
    
    # Apply envelope-related parameters when present
    if 'external_wall' in parameters:
        apply_construction_to_surfaces(idf, parameters['external_wall'], 'Wall')

    if 'roof' in parameters:
        apply_construction_to_surfaces(idf, parameters['roof'], 'Roof')

    if 'window' in parameters:
        apply_construction_to_surfaces(idf, parameters['window'], 'Window')

    # Apply infiltration
    if 'infiltration_ach' in parameters:
        apply_infiltration_to_zones(idf, parameters['infiltration_ach'])

    # Apply WWRs if any 'wwr_*' key is present (handles general and BR/LR-specific)
    if any(isinstance(k, str) and k.startswith('wwr_') for k in parameters.keys()):
        apply_wwr_to_windows(idf, parameters)

    # Apply albedo
    if 'wall_solar_absorptance' in parameters or 'roof_solar_absorptance' in parameters:
        albedo_dict = {}
        if 'wall_solar_absorptance' in parameters:
            albedo_dict["Cement/plaster/mortar - cement fibreboard magnesium oxysulphide binder_.0134"] = parameters['wall_solar_absorptance']
            albedo_dict["Cement/plaster/mortar - cement panels wood fibres_.0134"] = parameters['wall_solar_absorptance']
        if 'roof_solar_absorptance' in parameters:
            albedo_dict["Bitumen felt/sheet_.19"] = parameters['roof_solar_absorptance']
            albedo_dict["Bitumen felt/sheet_.003"] = parameters['roof_solar_absorptance']

        if albedo_dict:  # Only call if we have values to apply
            apply_solar_absorptance_to_surfaces(idf, albedo_dict)

    # Apply shading schedules
    if 'shading_schedule' in parameters:
        idf = apply_shading_schedules_to_idf(idf, parameters['shading_schedule'])

    # Create shading elements if provided as nested dimensions
    shading_dimensions = parameters.get('shading_dimensions')
    if shading_dimensions:
        # idf = create_shading_elements_by_orientation(idf, shading_dimensions)
        idf = create_shading_overhangs_and_fins_by_orientation(idf, shading_dimensions)

    # Save the modified IDF file if requested
    if save:
        save_path = save_path or idf.idfname
        idf.save(save_path)
    
    return idf


def run_simulation(
    idf_path, 
    weather_path, 
    params,
    idd_path=None, 
    output_dir=None, 
    verbose='v'
):
    """
    Run EnergyPlus simulation with envelope parameter modifications.
    
    Args:
        idf_path (str): Path to the IDF file
        weather_path (str): Path to the weather file (.epw)
        envelope_parameters (dict): Dictionary containing envelope parameters:
            - 'external_wall': External wall construction name
            - 'roof': Roof construction name
            - 'window': Window construction name
            - 'infiltration_ach': Infiltration rate in ACH at 50 Pa
            - 'wwr_east': Window-to-wall ratio for east facade
            - 'wwr_west': Window-to-wall ratio for west facade
            - 'wwr_north': Window-to-wall ratio for north facade
            - 'wwr_south': Window-to-wall ratio for south facade
        idd_path (str, optional): Path to the IDD file
        output_dir (str, optional): Directory for output files
        verbose (str, optional): Verbosity level for EnergyPlus output:
            - 'v': Verbose (default)
            - 'q': Quiet (suppresses most logging)
            - 's': Silent (minimal output)
    
    Returns:
        str: Output directory path where simulation results are stored
    """
    idf = None
    try:
        # Set up output directory with timestamp
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"simulation_output_{timestamp}"
        
        # Create multi_run directory if it doesn't exist
        multi_run_dir = "multi_run"
        os.makedirs(multi_run_dir, exist_ok=True)
        
        # Create the full output directory path within multi_run
        full_output_dir = os.path.join(multi_run_dir, output_dir)
        
        # Ensure output directory exists
        os.makedirs(full_output_dir, exist_ok=True)
        
        # Set the IDD file if provided
        if idd_path:
            IDF.setiddname(idd_path)
        
        # Initialize IDF with both IDF file and weather file
        idf = IDF(idf_path, weather_path)
        
        
        idf = modify_idf(idf, params)
        
        # Save the modified IDF file for later analysis
        modified_idf_path = os.path.join(full_output_dir, "modified_input.idf")
        idf.save(modified_idf_path)
        
        # Run simulation with specified output directory and verbosity
        idf.run(output_directory=full_output_dir, verbose=verbose)
        
        return full_output_dir
        
    finally:
        # Clear memory
        if idf is not None:
            del idf
        gc.collect()


def analyze_simulation_results(output_dir, cleanup=True, overheating_criterion='ASHRAE'):
    """
    Analyze simulation results from a completed EnergyPlus simulation.
    
    Args:
        output_dir (str): Path to the simulation output directory
        cleanup (bool): Whether to delete the output directory after analysis (default: True)
        overheating_criterion (str): Criterion for overheating analysis ('ASHRAE' or 'CIBSE59', default: 'ASHRAE')
    
    Returns:
        dict: Dictionary containing simulation results, district heating, and overheating analysis
    """
    try:
        # Extract district heating value
        htm_file_path = os.path.join(output_dir, "eplustbl.htm")
        if os.path.exists(htm_file_path):
            district_heating = extract_district_heating_from_htm(htm_file_path)
        else:
            district_heating = None
        
        # Extract temperature data for overheating analysis
        eso_file_path = os.path.join(output_dir, "eplusout.eso")
        if os.path.exists(eso_file_path):
            temperature_data, timestamps, variable_mapping = parse_eso_file(eso_file_path)
            zone_temperatures = extract_zone_temperatures(temperature_data, timestamps)
            
            # Perform overheating analysis based on criterion
            if overheating_criterion.upper() == 'CIBSE59':
                overheating_results = analyze_overheating_with_CIBSE59(zone_temperatures)
            else:
                overheating_results = analyze_overheating_with_ASHRAE(zone_temperatures)
            
            # Print overheating analysis results
            # print_overheating_analysis_results(overheating_results)
            
            average_overheating = overheating_results['average_overheating_hours']
            
            # Calculate overheating metrics based on criterion
            max_overheating_hours = 0.0
            max_living_room_overheating = 0.0
            max_bedroom_overheating = 0.0
            
            if overheating_results['zone_results']:
                if overheating_criterion.upper() == 'CIBSE59':
                    # For CIBSE59, calculate separate maximums for living rooms and bedrooms
                    living_room_overheating = []
                    bedroom_overheating = []
                    
                    for zone_name, zone_data in overheating_results['zone_results'].items():
                        if zone_data.get('zone_type') == 'bedroom':
                            bedroom_overheating.append(zone_data['overheating_hours'])
                        else:  # living_room/kitchen
                            living_room_overheating.append(zone_data['overheating_hours'])
                    
                    if living_room_overheating:
                        max_living_room_overheating = max(living_room_overheating)
                    if bedroom_overheating:
                        max_bedroom_overheating = max(bedroom_overheating)
                else:
                    # For ASHRAE, calculate overall maximum across all zones
                    max_overheating_hours = max(
                        zone_data['overheating_hours'] 
                        for zone_data in overheating_results['zone_results'].values()
                    )
        else:
            average_overheating = None
            max_overheating_hours = None
            max_living_room_overheating = None
            max_bedroom_overheating = None
            overheating_results = None
        
        # Calculate total shading area from the modified IDF file
        total_shading_area = 0.0
        modified_idf_path = os.path.join(output_dir, "modified_input.idf")
        if os.path.exists(modified_idf_path):
            try:
                # Load the modified IDF that was used for simulation
                idf = IDF(modified_idf_path)
                total_shading_area = calculate_total_shading_area(idf)
            except Exception as e:
                print(f"Warning: Could not calculate shading area: {e}")
                total_shading_area = 0.0
        
        results = {
            'success': True,
            'output_directory': output_dir,
            'district_heating_kwh': district_heating,
            'average_overheating_hours': average_overheating,
            'overheating_results': overheating_results,
            'overheating_criterion': overheating_criterion,
            'total_shading_area': total_shading_area
        }
        
        # Add criterion-specific results
        if overheating_criterion.upper() == 'CIBSE59':
            results['max_living_room_overheating'] = max_living_room_overheating
            results['max_bedroom_overheating'] = max_bedroom_overheating
        else:
            # For ASHRAE, include overall maximum overheating hours
            results['max_overheating_hours'] = max_overheating_hours
        
        # Clean up output directory if requested
        if cleanup and os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
                results['output_directory'] = None  # Indicate directory was cleaned up
                
                # Check if multi_run directory is now empty and remove it if so
                multi_run_dir = os.path.dirname(output_dir)
                if os.path.basename(multi_run_dir) == "multi_run" and os.path.exists(multi_run_dir):
                    try:
                        # Only remove if directory is empty
                        if not os.listdir(multi_run_dir):
                            os.rmdir(multi_run_dir)
                    except Exception:
                        # It's okay if we can't remove the multi_run directory
                        pass
                        
            except Exception as cleanup_error:
                # Log cleanup error but don't fail the analysis
                print("Warning: Failed to clean up output directory %s: %s" % (output_dir, cleanup_error))
        
        return results
        
    except Exception as e:
        # Clean up output directory even if analysis failed
        if cleanup and os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
                
                # Check if multi_run directory is now empty and remove it if so
                multi_run_dir = os.path.dirname(output_dir)
                if os.path.basename(multi_run_dir) == "multi_run" and os.path.exists(multi_run_dir):
                    try:
                        # Only remove if directory is empty
                        if not os.listdir(multi_run_dir):
                            os.rmdir(multi_run_dir)
                    except Exception:
                        # It's okay if we can't remove the multi_run directory
                        pass
                    
            except Exception as cleanup_error:
                print("Warning: Failed to clean up output directory %s: %s" % (output_dir, cleanup_error))
        
        return {
            'success': False,
            'error': str(e),
            'output_directory': None,
            'district_heating_kwh': None,
            'average_overheating_hours': None,
            'max_overheating_hours': None,
            'overheating_results': None,
            'overheating_criterion': overheating_criterion
        }


def run_energyplus_simulation_with_analysis(idf_path, weather_path, idd_path=None, output_dir=None, verbose='v'):
    """
    Legacy function - combines run_simulation and analyze_simulation_results.
    Use the separate functions for better memory management.
    
    Args:
        idf_path (str): Path to the IDF file
        weather_path (str): Path to the weather file (.epw)
        idd_path (str, optional): Path to the IDD file. If None, eppy will try to find it automatically.
        output_dir (str, optional): Directory for output files. If None, creates a timestamped folder.
        verbose (str, optional): Verbosity level for EnergyPlus output:
            - 'v': Verbose (default)
            - 'q': Quiet (suppresses most logging)
            - 's': Silent (minimal output)
    
    Returns:
        dict: Dictionary containing simulation results, district heating, and overheating analysis
    """
    # Create empty envelope parameters for backward compatibility
    envelope_parameters = {}
    
    # Run simulation
    output_dir = run_simulation(idf_path, weather_path, envelope_parameters, idd_path, output_dir, verbose)
    
    # Analyze results
    return analyze_simulation_results(output_dir)


def print_simulation_summary(results):
    """
    Print a summary of the simulation results.
    
    Args:
        results: Results dictionary from analyze_simulation_results
    """
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    
    if results['success']:
        print("✓ Simulation completed successfully")
        print("📁 Output directory: %s" % results['output_directory'])
        
        if results['district_heating_kwh'] is not None:
            print("🔥 District Heating: %.2f kWh" % results['district_heating_kwh'])
        else:
            print("❌ District Heating: Not available")
            
        if results['average_overheating_hours'] is not None:
            print("🌡️  Average Overheating: %.1f hours" % results['average_overheating_hours'])
        else:
            print("❌ Average Overheating: Not available")
    else:
        print("❌ Simulation failed: %s" % results.get('error', 'Unknown error'))


# Example usage with files in current directory
if __name__ == "__main__":
    idf_file = "./Building_with_temp_output.idf"
    weather_file = "./Vancouver - 2021 AMY.epw"
    idd_file = "C:/EnergyPlusV9-4-0/Energy+.idd"
    
    # Example envelope parameters
    envelope_params = {
        'external_wall': 'R-20 Wall',
        'roof': 'R-30 Roof',
        'window': 'Double Glazed',
        'infiltration_ach': 0.5,
        'wwr_east': 0.3,
        'wwr_west': 0.3,
        'wwr_north': 0.2,
        'wwr_south': 0.4
    }
    
    # Run simulation
    output_dir = run_simulation(idf_file, weather_file, envelope_params, idd_file)
    
    # Analyze results
    results = analyze_simulation_results(output_dir)
    
    # Print summary
    print_simulation_summary(results)


def _expand_partial_schedule(partial_values, time_slots_per_season, fixed_morning_values=None):
    """
    Expand partial schedule values to full schedule with optional fixed morning values.
    
    Args:
        partial_values: List of values for the time slots that are optimized
        time_slots_per_season: Number of time slots per season (e.g., 6 for full day)
        fixed_morning_values: List of fixed values for morning slots (e.g., [0, 0, 0] for 06-12)
                             If None, no fixed values are inserted
    
    Returns:
        List of full schedule values
    """
    if not partial_values:
        return [0] * (time_slots_per_season * 4)  # 4 seasons
    
    seasons = 4  # spring, summer, fall, winter
    values_per_season = len(partial_values) // seasons
    
    if len(partial_values) != values_per_season * seasons:
        raise ValueError(f"Expected {values_per_season * seasons} values, got {len(partial_values)}")
    
    full_schedule = []
    for season in range(seasons):
        season_start = season * values_per_season
        season_values = partial_values[season_start:season_start + values_per_season]
        
        if fixed_morning_values:
            # Insert fixed morning values at the beginning of each season
            full_schedule.extend(fixed_morning_values + season_values)
        else:
            # No fixed values, use as-is
            full_schedule.extend(season_values)
    
    return full_schedule


def build_seasonal_shading_schedules(optimization_variables, element_configs):
    """
    Build seasonal dict format for apply_shading_schedules_to_idf from optimization variables.
    
    Args:
        optimization_variables: List of optimization variables (0s and 1s)
        element_configs: List of dicts with element configuration:
            [
                {
                    'name': 'south_overhang',
                    'type': 'full_day',  # or 'afternoon_only'
                    'start_idx': 0,
                    'time_slots': 24,  # or 12 for afternoon_only
                    'fixed_morning': None  # or [0, 0, 0] for afternoon_only
                },
                ...
            ]
    
    Returns:
        Dict mapping element names to seasonal schedules
    """
    shading_dict = {}
    var_idx = 0
    
    for config in element_configs:
        element_name = config['name']
        element_type = config['type']
        start_idx = config['start_idx']
        # time_slots not directly used here; keep for API compatibility
        fixed_morning = config.get('fixed_morning')
        
        if element_type == 'full_day':
            # Take full 24 variables (4 seasons × 6 time slots)
            end_idx = start_idx + 24
            element_values = [int(v) for v in optimization_variables[start_idx:end_idx]]
            var_idx += 24
            
        elif element_type == 'afternoon_only':
            # Take 12 variables (4 seasons × 3 afternoon time slots)
            end_idx = start_idx + 12
            afternoon_values = [int(v) for v in optimization_variables[start_idx:end_idx]]
            # Expand to full 24-value schedule
            element_values = _expand_partial_schedule(
                afternoon_values, 
                time_slots_per_season=6, 
                fixed_morning_values=fixed_morning
            )
            var_idx += 12
        else:
            raise ValueError('Unknown element type: %s' % element_type)
        
        # Convert to seasonal dict format
        seasonal_dict = {}
        seasons = ['spring', 'summer', 'fall', 'winter']
        for i, season in enumerate(seasons):
            season_start = i * 6
            seasonal_dict[season] = element_values[season_start:season_start + 6]
        
        shading_dict[element_name] = seasonal_dict
    
    return shading_dict


def calculate_total_shading_area(idf) -> float:
    """Calculate the total surface area of all shading elements in the building.
    
    Args:
        idf: The IDF object containing shading elements
        
    Returns:
        Total shading area in square meters
    """
    total_area = 0.0
    
    # Calculate area for SHADING:BUILDING:DETAILED objects (overhangs and fins)
    for shading in idf.idfobjects.get('SHADING:BUILDING:DETAILED', []):
        try:
            # Get vertex coordinates
            vertices = []
            for i in range(1, 5):  # Assuming 4 vertices
                x = getattr(shading, f'Vertex_{i}_Xcoordinate', None)
                y = getattr(shading, f'Vertex_{i}_Ycoordinate', None)
                z = getattr(shading, f'Vertex_{i}_Zcoordinate', None)
                
                if x is not None and y is not None and z is not None:
                    try:
                        vertices.append((float(x), float(y), float(z)))
                    except (ValueError, TypeError):
                        continue
            
            if len(vertices) >= 3:
                # Calculate area using the shoelace formula for polygon area
                area = _calculate_polygon_area(vertices)
                total_area += area
                
        except (AttributeError, ValueError, TypeError):
            # Skip invalid shading elements
            continue
    
    # Get all fenestrations for reference
    fenestrations = idf.idfobjects.get('FENESTRATIONSURFACE:DETAILED', [])
    fenestration_dict = {fen.Name: fen for fen in fenestrations}
    
    # Calculate area for SHADING:OVERHANG objects
    for overhang in idf.idfobjects.get('SHADING:OVERHANG', []):
        window_name = getattr(overhang, 'Window_or_Door_Name', None)
        depth = getattr(overhang, 'Depth', 0.0)
        
        if window_name and window_name in fenestration_dict:
            window = fenestration_dict[window_name]
            window_width, window_height = _calculate_window_dimensions(window)
            
            # Overhang area = window width × depth
            overhang_area = window_width * depth
            total_area += overhang_area
    
    # Calculate area for SHADING:FIN objects
    for fin in idf.idfobjects.get('SHADING:FIN', []):
        window_name = getattr(fin, 'Window_or_Door_Name', None)
        left_depth = getattr(fin, 'Left_Depth', 0.0)
        right_depth = getattr(fin, 'Right_Depth', 0.0)
        
        if window_name and window_name in fenestration_dict:
            window = fenestration_dict[window_name]
            window_width, window_height = _calculate_window_dimensions(window)
            
            # Convert depth values to float, handling empty strings and invalid values
            try:
                left_depth_float = float(left_depth) if left_depth and str(left_depth).strip() else 0.0
                right_depth_float = float(right_depth) if right_depth and str(right_depth).strip() else 0.0
            except (ValueError, TypeError):
                left_depth_float = 0.0
                right_depth_float = 0.0
            
            # Fin areas = window height × depth (left and right fins)
            left_fin_area = window_height * left_depth_float
            right_fin_area = window_height * right_depth_float
            total_area += left_fin_area + right_fin_area
    
    return total_area


def _calculate_window_dimensions(fenestration) -> tuple[float, float]:
    """Calculate the width and height of a fenestration surface.
    
    Args:
        fenestration: FENESTRATIONSURFACE:DETAILED object
        
    Returns:
        tuple: (width, height) in meters
    """
    vertices = []
    for i in range(1, 5):  # Assuming 4 vertices
        x = getattr(fenestration, f'Vertex_{i}_Xcoordinate', None)
        y = getattr(fenestration, f'Vertex_{i}_Ycoordinate', None)
        z = getattr(fenestration, f'Vertex_{i}_Zcoordinate', None)
        
        if x is not None and y is not None and z is not None:
            try:
                vertices.append((float(x), float(y), float(z)))
            except (ValueError, TypeError):
                continue
    
    if len(vertices) < 4:
        return 0.0, 0.0
    
    # Calculate width and height from vertices
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    
    # Width is the range in X or Y (whichever varies more)
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    z_range = max(zs) - min(zs)
    
    # Height is always the Z range
    height = z_range
    
    # Width is the larger of X or Y range (assuming window is oriented along one axis)
    width = max(x_range, y_range)
    
    return width, height


def _calculate_polygon_area(vertices):
    """Calculate the area of a polygon using the shoelace formula.
    
    Args:
        vertices: List of (x, y, z) tuples representing polygon vertices
        
    Returns:
        Area in square meters
    """
    if len(vertices) < 3:
        return 0.0
    
    # For 3D polygons, we need to determine which plane they're in
    # and project onto that plane to calculate the area
    
    # Check if the polygon is in the XY, XZ, or YZ plane
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    z_range = max(zs) - min(zs)
    
    # Determine which plane the polygon is in based on which coordinate varies least
    if x_range < 0.001:  # Polygon is in YZ plane (constant X)
        coords = [(v[1], v[2]) for v in vertices]
    elif y_range < 0.001:  # Polygon is in XZ plane (constant Y)
        coords = [(v[0], v[2]) for v in vertices]
    elif z_range < 0.001:  # Polygon is in XY plane (constant Z)
        coords = [(v[0], v[1]) for v in vertices]
    else:
        # Polygon is not aligned with any coordinate plane
        # Use XY projection as fallback
        coords = [(v[0], v[1]) for v in vertices]
    
    # Use the shoelace formula to calculate area
    n = len(coords)
    area = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    
    return abs(area) / 2.0


def _compute_overhang_extensions(
    fen_vertices: List[Tuple[float, float, float]],
    orientation: str,
    params: Dict[str, Any],
) -> Tuple[float, float]:
    """Compute left/right overhang extensions in meters.

    Priority:
    1) Use explicit percentage fields 'left_extension_pct'/'right_extension_pct'
       of the window span along the lateral axis for the orientation.
    2) Fallback to explicit meter fields 'left_extension'/'right_extension' if present.
    3) Fallback to legacy 'width' relative factor -> symmetric extra split.
    """
    xs = [v[0] for v in fen_vertices]
    ys = [v[1] for v in fen_vertices]
    span_x = (max(xs) - min(xs)) if xs else 0.0
    span_y = (max(ys) - min(ys)) if ys else 0.0
    window_span = span_x if orientation == 'south' else span_y

    # 1) Percent-based
    l_pct = params.get('left_extension_pct', None)
    r_pct = params.get('right_extension_pct', None)

    try:
        l_pct_f = float(l_pct) if l_pct is not None else None
    except (TypeError, ValueError):
        l_pct_f = None

    try:
        r_pct_f = float(r_pct) if r_pct is not None else None
    except (TypeError, ValueError):
        r_pct_f = None

    if l_pct_f is not None or r_pct_f is not None:
        left = max(0.0, (l_pct_f or 0.0) * window_span)
        right = max(0.0, (r_pct_f or 0.0) * window_span)
        return left, right

    # 2) Explicit meters
    try:
        left_ext = float(params.get('left_extension', 0.0) or 0.0)
    except (TypeError, ValueError):
        left_ext = 0.0

    try:
        right_ext = float(params.get('right_extension', 0.0) or 0.0)
    except (TypeError, ValueError):
        right_ext = 0.0

    if left_ext > 0.0 or right_ext > 0.0:
        return max(0.0, left_ext), max(0.0, right_ext)

    # 3) Legacy width factor -> symmetric extra coverage
    try:
        rel_width = float(params.get('width', 1.0) or 1.0)
    except (TypeError, ValueError):
        rel_width = 1.0

    extra = max(window_span * rel_width - window_span, 0.0)
    half = 0.5 * extra
    return half, half


def _create_overhang_parametric(
    idf: IDF,
    fenestration: Any,
    fen_vertices: List[Tuple[float, float, float]],
    params: Dict[str, Any],
    orientation: str,
) -> None:
    """Create a SHADING:OVERHANG object for a fenestration.

    Supported params: depth, angle, height_above, left_extension, right_extension,
    and legacy width.
    """
    if not params or 'depth' not in params:
        return
    depth = float(params.get('depth', 0.0) or 0.0)
    angle = float(params.get('angle', 0.0) or 0.0)
    height_above = float(params.get('height_above', 0.0) or 0.0)
    left_ext, right_ext = _compute_overhang_extensions(fenestration and [(getattr(fenestration, f'Vertex_{i}_Xcoordinate', 0.0) or 0.0, getattr(fenestration, f'Vertex_{i}_Ycoordinate', 0.0) or 0.0, getattr(fenestration, f'Vertex_{i}_Zcoordinate', 0.0) or 0.0) for i in range(1,5)] or [], orientation, params)

    ov = idf.newidfobject('SHADING:OVERHANG')
    ov.Name = f"{orientation.capitalize()} Overhang {fenestration.Name}"
    ov.Window_or_Door_Name = fenestration.Name
    ov.Height_above_Window_or_Door = height_above
    ov.Tilt_Angle_from_WindowDoor = angle
    ov.Left_extension_from_WindowDoor_Width = left_ext
    ov.Right_extension_from_WindowDoor_Width = right_ext
    ov.Depth = depth


def _create_fin_parametric(
    idf: IDF,
    fenestration: Any,
    side_params: Dict[str, Any],
    other_side_params: Dict[str, Any],
    orientation: str,
) -> None:
    """Create a SHADING:FIN object for a fenestration.

    Each side may specify: depth, extension, distance_above_top,
    distance_below_bottom, tilt (all floats).
    """
    if not side_params and not other_side_params:
        return

    left_depth = float((side_params or {}).get('depth', 0.0) or 0.0)
    right_depth = float((other_side_params or {}).get('depth', 0.0) or 0.0)
    left_ext = float((side_params or {}).get('extension', 0.0) or 0.0)
    right_ext = float((other_side_params or {}).get('extension', 0.0) or 0.0)
    left_above = float((side_params or {}).get('distance_above_top', 0.0) or 0.0)
    right_above = float((other_side_params or {}).get('distance_above_top', 0.0) or 0.0)
    left_below = float((side_params or {}).get('distance_below_bottom', 0.0) or 0.0)
    right_below = float((other_side_params or {}).get('distance_below_bottom', 0.0) or 0.0)
    left_tilt = float((side_params or {}).get('tilt', 90.0) or 90.0)
    right_tilt = float((other_side_params or {}).get('tilt', 90.0) or 90.0)

    fin = idf.newidfobject('SHADING:FIN')
    fin.Name = f"{orientation.capitalize()} Fin {fenestration.Name}"
    fin.Window_or_Door_Name = fenestration.Name
    
    # Left side
    fin.Left_Extension_from_WindowDoor = left_ext
    fin.Left_Distance_Above_Top_of_Window = left_above
    fin.Left_Distance_Below_Bottom_of_Window = left_below
    fin.Left_Tilt_Angle_from_WindowDoor = left_tilt
    fin.Left_Depth = left_depth
    
    # Right side
    fin.Right_Extension_from_WindowDoor = right_ext
    fin.Right_Distance_Above_Top_of_Window = right_above
    fin.Right_Distance_Below_Bottom_of_Window = right_below
    fin.Right_Tilt_Angle_from_WindowDoor = right_tilt
    fin.Right_Depth = right_depth


def identify_wall_orientation_by_vertices(surface) -> Optional[str]:
    """Identify wall orientation by analyzing vertex coordinates.

    EnergyPlus vertices are ordered counter-clockwise starting from the bottom left corner.
    For North/South walls (constant Y): analyze X coordinate change between consecutive vertices
    For East/West walls (constant X): analyze Y coordinate change between consecutive vertices

    Args:
        surface: Building surface object with vertex coordinates

    Returns:
        str: Orientation ('North', 'South', 'East', 'West') or None if cannot determine
    """
    vertices = []
    for i in range(1, 5):
        x = getattr(surface, f'Vertex_{i}_Xcoordinate', None)
        y = getattr(surface, f'Vertex_{i}_Ycoordinate', None)
        z = getattr(surface, f'Vertex_{i}_Zcoordinate', None)
        if x is not None and y is not None and z is not None:
            vertices.append((float(x), float(y), float(z)))

    if len(vertices) != 4:
        return None

    y_coords = [v[1] for v in vertices]
    x_coords = [v[0] for v in vertices]

    if all(abs(y - y_coords[0]) < 1e-3 for y in y_coords):
        x_change = vertices[1][0] - vertices[0][0]
        if abs(x_change) > 1e-3:
            return 'South' if x_change > 0 else 'North'

    elif all(abs(x - x_coords[0]) < 1e-3 for x in x_coords):
        y_change = vertices[1][1] - vertices[0][1]
        if abs(y_change) > 1e-3:
            return 'East' if y_change > 0 else 'West'

    return None


def create_shading_overhangs_and_fins_by_orientation(
    idf: IDF, dimensions: Optional[Dict[str, Any]]
):
    """Create SHADING:OVERHANG and SHADING:FIN objects by wall orientation.

    Accepts the same arguments as create_shading_elements_by_orientation.
    """
    if not dimensions:
        return idf

    fenestrations = idf.idfobjects.get('FENESTRATIONSURFACE:DETAILED', [])
    for fen in fenestrations:
        parent_wall_name = getattr(fen, 'Building_Surface_Name', None)
        parent_wall = None

        for wall in idf.idfobjects.get('BUILDINGSURFACE:DETAILED', []):
            if wall.Name == parent_wall_name:
                parent_wall = wall
                break

        if parent_wall is None:
            continue

        orientation = identify_wall_orientation_by_vertices(parent_wall)

        if orientation is None:
            continue

        orientation = orientation.lower()

        fen_vertices = []
        for i in range(1,5):
            x = getattr(fen, f'Vertex_{i}_Xcoordinate', None)
            y = getattr(fen, f'Vertex_{i}_Ycoordinate', None)
            z = getattr(fen, f'Vertex_{i}_Zcoordinate', None)
            if x is not None and y is not None and z is not None:
                fen_vertices.append((float(x), float(y), float(z)))
        if not fen_vertices:
            continue

        if orientation not in dimensions:
            continue

        side_dims = dimensions.get(orientation, {}) or {}

        if isinstance(side_dims.get('overhang'), dict):
            _create_overhang_parametric(
                idf,
                fen,
                fen_vertices,
                side_dims['overhang'],
                orientation,
            )

        left = side_dims.get('left_fin') if isinstance(side_dims.get('left_fin'), dict) else None
        right = side_dims.get('right_fin') if isinstance(side_dims.get('right_fin'), dict) else None

        if left or right:
            _create_fin_parametric(
                idf,
                fen,
                left or {},
                right or {},
                orientation
            )

    return idf

def _copy_idf_object(dst: IDF, obj: Any, key_hint: Optional[str] = None):
    """Copy an IDF object into dst. Fallback to creating new and copying attrs.

    Args:
        dst: Destination IDF
        obj: Source object instance
        key_hint: Optional IDD object key when copy API not available
    Returns:
        The new object in dst, or None on failure
    """
    try:
        return dst.copyidfobject(obj)  # type: ignore[attr-defined]
    except Exception:
        key = key_hint or getattr(obj, 'key', None) or getattr(obj, 'objls', [None])[0]
        if not key:
            return None
        new_obj = dst.newidfobject(str(key))
        for attr in dir(obj):
            if attr.startswith('_'):
                continue
            try:
                val = getattr(obj, attr)
            except Exception:
                continue
            if callable(val):
                continue
            try:
                setattr(new_obj, attr, val)
            except Exception:
                pass
        return new_obj

def bulk_copy_window_related_objects(src: IDF, dst: IDF) -> Dict[str, int]:
    """Copy all window-related objects by class if missing in destination.

    Classes copied:
    - CONSTRUCTION
    - MATERIALPROPERTY:GLAZINGSPECTRALDATA
    - All classes starting with WINDOWMATERIAL:

    Objects are copied by original name only if that name doesn't already
    exist in the destination for the same class. No renaming or retargeting
    is performed. This keeps behavior simple and predictable.

    Returns a dict of class name -> number of objects copied.
    """
    if not src or not dst:
        return {}

    classes: Set[str] = set()
    classes.add('CONSTRUCTION')
    # Base material classes
    classes.add('MATERIAL')
    classes.add('MATERIAL:NOMASS')
    classes.add('MATERIAL:INFRAREDTRANSPARENT')
    # Include all MaterialProperty:Glazing* classes (e.g., GlazingSpectralData)
    for key in src.idfobjects.keys():
        ukey = str(key).upper()
        if ukey.startswith('MATERIALPROPERTY:GLAZING'):
            classes.add(str(key))
    # Include all WindowMaterial:* classes
    for key in src.idfobjects.keys():
        ukey = str(key).upper()
        if ukey.startswith('WINDOWMATERIAL:'):
            classes.add(str(key))
    # Include all WindowProperty:* classes (e.g., FrameAndDivider, ShadingControl)
    for key in src.idfobjects.keys():
        ukey = str(key).upper()
        if ukey.startswith('WINDOWPROPERTY:'):
            classes.add(str(key))

    copied_counts: Dict[str, int] = {cls: 0 for cls in classes}

    for cls in sorted(classes):
        src_objs = src.idfobjects.get(cls, [])
        dst_objs = dst.idfobjects.get(cls, [])
        dst_names = {
            str(getattr(o, 'Name', '')).strip() for o in dst_objs if hasattr(o, 'Name')
        }

        for obj in src_objs:
            name = str(getattr(obj, 'Name', '')).strip()
            if not name or name in dst_names:
                continue
            new_obj = _copy_idf_object(dst, obj, key_hint=cls)
            if new_obj is None:
                continue
            try:
                new_obj.Name = name
            except Exception:
                pass
            dst_names.add(name)
            copied_counts[cls] += 1

    # Remove classes with zero copies for a cleaner report
    return {k: v for k, v in copied_counts.items() if v > 0}


def bulk_copy_window_related_objects_files(
    src_path: str,
    dst_path: str,
    idd_path: Optional[str] = None,
    save_as: Optional[str] = None,
) -> Dict[str, int]:
    """File-based wrapper for bulk copying window-related objects.

    Copies by original names, only when missing in destination.
    Saves to save_as if provided, else overwrites destination.
    """
    if idd_path:
        IDF.setiddname(idd_path)

    src_idf = IDF(src_path)
    dst_idf = IDF(dst_path)

    report = bulk_copy_window_related_objects(src_idf, dst_idf)

    if save_as:
        dst_idf.save(save_as)
    else:
        dst_idf.save()

    return report



def copy_window_constructions_and_materials(
    src: IDF,
    dst: IDF,
    suffix: str = ''
) -> Dict[str, str]:
    """
    Copy window constructions referenced by windows in src to dst with a name suffix.

    - Copies the CONSTRUCTION and any referenced WINDOWMATERIAL:* or MATERIAL:* layers.
    - Copies referenced MATERIALPROPERTY:GLAZINGSPECTRALDATA if present and rewires glazing.
    - Returns mapping of original construction name -> new suffixed name.
    """
    mapping: Dict[str, str] = {}
    if not src or not dst:
        return mapping

    windows = src.idfobjects.get('FENESTRATIONSURFACE:DETAILED', [])
    if not windows:
        return mapping

    # Gather constructions used by windows
    used_constructions: Set[str] = set()
    for win in windows:
        name = getattr(win, 'Construction_Name', None)
        if name:
            used_constructions.add(str(name))

    if not used_constructions:
        return mapping

    # Index source objects by name for quick lookup
    src_constructions = {str(getattr(c, 'Name', '')): c for c in src.idfobjects.get('CONSTRUCTION', [])}
    src_glazings = {str(getattr(m, 'Name', '')): m for m in src.idfobjects.get('WINDOWMATERIAL:GLAZING', [])}
    src_gases = {str(getattr(m, 'Name', '')): m for m in src.idfobjects.get('WINDOWMATERIAL:GAS', [])}
    src_materials = {str(getattr(m, 'Name', '')): m for m in src.idfobjects.get('MATERIAL', [])}
    src_spectral = {str(getattr(s, 'Name', '')): s for s in src.idfobjects.get('MATERIALPROPERTY:GLAZINGSPECTRALDATA', [])}

    # Helper to clone an object into dst with a new name
    def _clone_obj(obj: Any, key: str, new_name: str) -> Any:
        try:
            new_obj = dst.newidfobject(key)
        except Exception:
            return None
        # Copy all public attributes
        for attr in dir(obj):
            if attr.startswith('_'):
                continue
            try:
                val = getattr(obj, attr)
            except Exception:
                continue
            if callable(val):
                continue
            try:
                setattr(new_obj, attr, val)
            except Exception:
                pass
        # Set new name if applicable
        try:
            new_obj.Name = new_name
        except Exception:
            pass
        return new_obj

    for cons_name in used_constructions:
        cons = src_constructions.get(cons_name)
        if not cons:
            continue
        new_cons_name = f"{cons_name}{suffix}" if suffix else cons_name

        # Create destination CONSTRUCTION
        new_cons = _clone_obj(cons, 'CONSTRUCTION', new_cons_name)
        if not new_cons:
            continue

        mapping[cons_name] = new_cons_name

        # Copy up to, say, 10 layers heuristically (Outside_Layer, Layer_2, ... Layer_10)
        layer_attrs = ['Outside_Layer'] + [f'Layer_{i}' for i in range(2, 11)]
        for attr in layer_attrs:
            mat_name = getattr(cons, attr, None)
            if not mat_name:
                continue
            mat_name = str(mat_name)
            new_mat_name = f"{mat_name}{suffix}" if suffix else mat_name

            # Glazing
            if mat_name in src_glazings:
                glz = src_glazings[mat_name]
                new_glz = _clone_obj(glz, 'WINDOWMATERIAL:GLAZING', new_mat_name)
                # Rewire spectral set if present
                try:
                    spec_name = getattr(glz, 'Spectral_Data_Set_Name')
                except Exception:
                    spec_name = None
                if spec_name:
                    spec_name = str(spec_name)
                    if spec_name in src_spectral:
                        new_spec_name = f"{spec_name}{suffix}" if suffix else spec_name
                        _clone_obj(src_spectral[spec_name], 'MATERIALPROPERTY:GLAZINGSPECTRALDATA', new_spec_name)
                        try:
                            new_glz.Spectral_Data_Set_Name = new_spec_name
                        except Exception:
                            pass

            # Gas layer
            elif mat_name in src_gases:
                _clone_obj(src_gases[mat_name], 'WINDOWMATERIAL:GAS', new_mat_name)

            # Opaque material layer (rare in window stacks but safe)
            elif mat_name in src_materials:
                _clone_obj(src_materials[mat_name], 'MATERIAL', new_mat_name)

            # Update layer reference in the new construction
            try:
                setattr(new_cons, attr, new_mat_name)
            except Exception:
                pass

    return mapping

