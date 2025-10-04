"""
Geometric Modeling Kernel for OmniCAD

This module provides the core geometric modeling capabilities including:
- BREP (Boundary Representation) modeling
- NURBS curves and surfaces
- Solid modeling operations
- Computational geometry algorithms
- Feature-based parametric modeling

The kernel is designed to be extensible and can integrate with external
geometry libraries like OpenCASCADE or FreeCAD's geometric kernel.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import math

from ..utils.logger import OmniLogger
from ..utils.event_system import EventSystem


class GeometryType(Enum):
    """Types of geometric entities"""
    POINT = "point"
    CURVE = "curve"
    SURFACE = "surface"
    SOLID = "solid"
    ASSEMBLY = "assembly"
    
    # Curve types
    LINE = "line"
    CIRCLE = "circle"
    ARC = "arc"
    ELLIPSE = "ellipse"
    SPLINE = "spline"
    NURBS_CURVE = "nurbs_curve"
    
    # Surface types
    PLANE = "plane"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    CONE = "cone"
    TORUS = "torus"
    NURBS_SURFACE = "nurbs_surface"
    
    # Solid types
    BOX = "box"
    CYLINDER_SOLID = "cylinder_solid"
    SPHERE_SOLID = "sphere_solid"
    CONE_SOLID = "cone_solid"
    TORUS_SOLID = "torus_solid"
    EXTRUDE = "extrude"
    REVOLVE = "revolve"
    SWEEP = "sweep"
    LOFT = "loft"


@dataclass
class Transform3D:
    """3D transformation matrix"""
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4))
    
    @classmethod
    def identity(cls):
        """Create identity transform"""
        return cls()
    
    @classmethod
    def translation(cls, x: float, y: float, z: float):
        """Create translation transform"""
        matrix = np.eye(4)
        matrix[0:3, 3] = [x, y, z]
        return cls(matrix)
    
    @classmethod
    def rotation_x(cls, angle: float):
        """Create rotation around X axis"""
        matrix = np.eye(4)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        matrix[1, 1] = cos_a
        matrix[1, 2] = -sin_a
        matrix[2, 1] = sin_a
        matrix[2, 2] = cos_a
        return cls(matrix)
    
    @classmethod
    def rotation_y(cls, angle: float):
        """Create rotation around Y axis"""
        matrix = np.eye(4)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        matrix[0, 0] = cos_a
        matrix[0, 2] = sin_a
        matrix[2, 0] = -sin_a
        matrix[2, 2] = cos_a
        return cls(matrix)
    
    @classmethod
    def rotation_z(cls, angle: float):
        """Create rotation around Z axis"""
        matrix = np.eye(4)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        matrix[0, 0] = cos_a
        matrix[0, 1] = -sin_a
        matrix[1, 0] = sin_a
        matrix[1, 1] = cos_a
        return cls(matrix)
    
    @classmethod
    def scale(cls, sx: float, sy: float, sz: float):
        """Create scale transform"""
        matrix = np.eye(4)
        matrix[0, 0] = sx
        matrix[1, 1] = sy
        matrix[2, 2] = sz
        return cls(matrix)
    
    def compose(self, other: 'Transform3D') -> 'Transform3D':
        """Compose with another transform"""
        return Transform3D(np.dot(self.matrix, other.matrix))
    
    def inverse(self) -> 'Transform3D':
        """Get inverse transform"""
        return Transform3D(np.linalg.inv(self.matrix))
    
    def apply_to_point(self, point: np.ndarray) -> np.ndarray:
        """Apply transform to a point"""
        if point.shape == (3,):
            point_4d = np.append(point, 1.0)
        else:
            point_4d = point
        
        result = np.dot(self.matrix, point_4d)
        return result[:3] / result[3]
    
    def apply_to_vector(self, vector: np.ndarray) -> np.ndarray:
        """Apply transform to a vector (no translation)"""
        vector_4d = np.append(vector, 0.0)
        result = np.dot(self.matrix, vector_4d)
        return result[:3]


@dataclass
class BoundingBox:
    """3D bounding box"""
    min_point: np.ndarray = field(default_factory=lambda: np.array([float('inf')] * 3))
    max_point: np.ndarray = field(default_factory=lambda: np.array([float('-inf')] * 3))
    
    def is_valid(self) -> bool:
        """Check if bounding box is valid"""
        return np.all(self.min_point <= self.max_point)
    
    def expand(self, point: np.ndarray):
        """Expand bounding box to include point"""
        self.min_point = np.minimum(self.min_point, point)
        self.max_point = np.maximum(self.max_point, point)
    
    def union(self, other: 'BoundingBox') -> 'BoundingBox':
        """Union with another bounding box"""
        if not other.is_valid():
            return self
        if not self.is_valid():
            return other
        
        return BoundingBox(
            np.minimum(self.min_point, other.min_point),
            np.maximum(self.max_point, other.max_point)
        )
    
    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if intersects with another bounding box"""
        return np.all(self.min_point <= other.max_point) and np.all(other.min_point <= self.max_point)
    
    def center(self) -> np.ndarray:
        """Get center point"""
        return (self.min_point + self.max_point) / 2
    
    def size(self) -> np.ndarray:
        """Get size in each dimension"""
        return self.max_point - self.min_point
    
    def volume(self) -> float:
        """Get volume"""
        if not self.is_valid():
            return 0.0
        size = self.size()
        return size[0] * size[1] * size[2]


class GeometryEntity:
    """
    Base class for all geometric entities in OmniCAD.
    
    This provides the foundation for BREP modeling with parametric capabilities.
    """
    
    def __init__(self, geometry_type: GeometryType, parameters: Dict[str, Any] = None):
        self.id = str(uuid.uuid4())
        self.geometry_type = geometry_type
        self.parameters = parameters or {}
        self.transform = Transform3D.identity()
        self.bounding_box = BoundingBox()
        
        # Parametric modeling support
        self.is_parametric = True
        self.parameter_constraints = {}
        self.dependency_graph = {}
        
        # Visualization data
        self.tessellation_data = None
        self.visual_properties = {
            'color': [0.7, 0.7, 0.7, 1.0],
            'wireframe': False,
            'visible': True,
            'transparency': 0.0
        }
        
        # Validation state
        self.is_valid = True
        self.validation_errors = []
    
    def update_parameters(self, **kwargs):
        """Update parameters and regenerate geometry"""
        self.parameters.update(kwargs)
        self.regenerate()
    
    def regenerate(self):
        """Regenerate geometry from parameters"""
        self._compute_geometry()
        self._update_bounding_box()
        self._invalidate_tessellation()
    
    def _compute_geometry(self):
        """Compute the actual geometry (override in subclasses)"""
        pass
    
    def _update_bounding_box(self):
        """Update bounding box (override in subclasses)"""
        pass
    
    def _invalidate_tessellation(self):
        """Invalidate cached tessellation data"""
        self.tessellation_data = None
    
    def get_tessellation(self, tolerance: float = 0.1) -> Dict[str, Any]:
        """Get tessellation data for rendering"""
        if self.tessellation_data is None:
            self.tessellation_data = self._generate_tessellation(tolerance)
        return self.tessellation_data
    
    def _generate_tessellation(self, tolerance: float) -> Dict[str, Any]:
        """Generate tessellation data (override in subclasses)"""
        return {
            'vertices': np.array([]),
            'indices': np.array([]),
            'normals': np.array([]),
            'uvs': np.array([])
        }
    
    def set_transform(self, transform: Transform3D):
        """Set transformation"""
        self.transform = transform
        self._update_bounding_box()
        self._invalidate_tessellation()
    
    def get_world_bounding_box(self) -> BoundingBox:
        """Get bounding box in world coordinates"""
        if not self.bounding_box.is_valid():
            return self.bounding_box
        
        # Transform bounding box corners
        corners = [
            np.array([self.bounding_box.min_point[0], self.bounding_box.min_point[1], self.bounding_box.min_point[2]]),
            np.array([self.bounding_box.min_point[0], self.bounding_box.min_point[1], self.bounding_box.max_point[2]]),
            np.array([self.bounding_box.min_point[0], self.bounding_box.max_point[1], self.bounding_box.min_point[2]]),
            np.array([self.bounding_box.min_point[0], self.bounding_box.max_point[1], self.bounding_box.max_point[2]]),
            np.array([self.bounding_box.max_point[0], self.bounding_box.min_point[1], self.bounding_box.min_point[2]]),
            np.array([self.bounding_box.max_point[0], self.bounding_box.min_point[1], self.bounding_box.max_point[2]]),
            np.array([self.bounding_box.max_point[0], self.bounding_box.max_point[1], self.bounding_box.min_point[2]]),
            np.array([self.bounding_box.max_point[0], self.bounding_box.max_point[1], self.bounding_box.max_point[2]])
        ]
        
        world_bbox = BoundingBox()
        for corner in corners:
            world_corner = self.transform.apply_to_point(corner)
            world_bbox.expand(world_corner)
        
        return world_bbox
    
    def validate(self) -> bool:
        """Validate geometry"""
        self.validation_errors.clear()
        self.is_valid = self._validate_geometry()
        return self.is_valid
    
    def _validate_geometry(self) -> bool:
        """Validate geometry implementation (override in subclasses)"""
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'type': self.geometry_type.value,
            'parameters': self.parameters,
            'transform': self.transform.matrix.tolist(),
            'visual_properties': self.visual_properties,
            'is_parametric': self.is_parametric
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeometryEntity':
        """Create from dictionary"""
        geometry_type = GeometryType(data['type'])
        entity = cls(geometry_type, data.get('parameters', {}))
        
        entity.id = data['id']
        entity.transform = Transform3D(np.array(data['transform']))
        entity.visual_properties = data.get('visual_properties', entity.visual_properties)
        entity.is_parametric = data.get('is_parametric', True)
        
        entity.regenerate()
        return entity


class Point(GeometryEntity):
    """3D Point geometry"""
    
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        super().__init__(GeometryType.POINT, {
            'x': x, 'y': y, 'z': z
        })
        self.regenerate()
    
    def _compute_geometry(self):
        """Compute point geometry"""
        self.point = np.array([
            self.parameters['x'],
            self.parameters['y'],
            self.parameters['z']
        ])
    
    def _update_bounding_box(self):
        """Update bounding box"""
        world_point = self.transform.apply_to_point(self.point)
        self.bounding_box = BoundingBox(world_point.copy(), world_point.copy())
    
    def _generate_tessellation(self, tolerance: float) -> Dict[str, Any]:
        """Generate point tessellation"""
        world_point = self.transform.apply_to_point(self.point)
        return {
            'vertices': world_point.reshape(1, 3),
            'indices': np.array([0]),
            'normals': np.array([[0, 0, 1]]),
            'uvs': np.array([[0, 0]])
        }


class Line(GeometryEntity):
    """3D Line geometry"""
    
    def __init__(self, start_point: np.ndarray, end_point: np.ndarray):
        super().__init__(GeometryType.LINE, {
            'start': start_point.tolist(),
            'end': end_point.tolist()
        })
        self.regenerate()
    
    def _compute_geometry(self):
        """Compute line geometry"""
        self.start_point = np.array(self.parameters['start'])
        self.end_point = np.array(self.parameters['end'])
        self.direction = self.end_point - self.start_point
        self.length = np.linalg.norm(self.direction)
        if self.length > 0:
            self.unit_direction = self.direction / self.length
        else:
            self.unit_direction = np.array([1, 0, 0])
    
    def _update_bounding_box(self):
        """Update bounding box"""
        world_start = self.transform.apply_to_point(self.start_point)
        world_end = self.transform.apply_to_point(self.end_point)
        
        self.bounding_box = BoundingBox()
        self.bounding_box.expand(world_start)
        self.bounding_box.expand(world_end)
    
    def _generate_tessellation(self, tolerance: float) -> Dict[str, Any]:
        """Generate line tessellation"""
        world_start = self.transform.apply_to_point(self.start_point)
        world_end = self.transform.apply_to_point(self.end_point)
        
        return {
            'vertices': np.array([world_start, world_end]),
            'indices': np.array([0, 1]),
            'normals': np.array([[0, 0, 1], [0, 0, 1]]),
            'uvs': np.array([[0, 0], [1, 0]])
        }
    
    def point_at_parameter(self, t: float) -> np.ndarray:
        """Get point at parameter t (0 to 1)"""
        return self.start_point + t * self.direction


class Circle(GeometryEntity):
    """3D Circle geometry"""
    
    def __init__(self, center: np.ndarray, radius: float, normal: np.ndarray = None):
        if normal is None:
            normal = np.array([0, 0, 1])
        
        super().__init__(GeometryType.CIRCLE, {
            'center': center.tolist(),
            'radius': radius,
            'normal': normal.tolist()
        })
        self.regenerate()
    
    def _compute_geometry(self):
        """Compute circle geometry"""
        self.center = np.array(self.parameters['center'])
        self.radius = self.parameters['radius']
        self.normal = np.array(self.parameters['normal'])
        self.normal = self.normal / np.linalg.norm(self.normal)
        
        # Create local coordinate system
        if abs(np.dot(self.normal, np.array([1, 0, 0]))) < 0.9:
            self.u_axis = np.cross(self.normal, np.array([1, 0, 0]))
        else:
            self.u_axis = np.cross(self.normal, np.array([0, 1, 0]))
        
        self.u_axis = self.u_axis / np.linalg.norm(self.u_axis)
        self.v_axis = np.cross(self.normal, self.u_axis)
    
    def _update_bounding_box(self):
        """Update bounding box"""
        # Create bounding box from circle extents
        extent = self.radius * np.sqrt(2)  # Conservative estimate
        min_point = self.center - extent
        max_point = self.center + extent
        
        self.bounding_box = BoundingBox(min_point, max_point)
    
    def _generate_tessellation(self, tolerance: float) -> Dict[str, Any]:
        """Generate circle tessellation"""
        # Calculate number of segments based on tolerance
        segments = max(8, int(2 * math.pi * self.radius / tolerance))
        
        vertices = []
        indices = []
        normals = []
        uvs = []
        
        # Generate circle points
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            
            # Local circle point
            local_point = self.center + self.radius * (cos_a * self.u_axis + sin_a * self.v_axis)
            
            # Transform to world space
            world_point = self.transform.apply_to_point(local_point)
            world_normal = self.transform.apply_to_vector(self.normal)
            
            vertices.append(world_point)
            normals.append(world_normal)
            uvs.append([cos_a * 0.5 + 0.5, sin_a * 0.5 + 0.5])
            
            # Line indices for wireframe
            indices.extend([i, (i + 1) % segments])
        
        return {
            'vertices': np.array(vertices),
            'indices': np.array(indices),
            'normals': np.array(normals),
            'uvs': np.array(uvs)
        }
    
    def point_at_angle(self, angle: float) -> np.ndarray:
        """Get point at given angle"""
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return self.center + self.radius * (cos_a * self.u_axis + sin_a * self.v_axis)


class GeometryKernel:
    """
    Main geometry kernel for OmniCAD.
    
    Provides high-level geometric operations and manages the geometry database.
    """
    
    def __init__(self):
        self.logger = OmniLogger("GeometryKernel")
        self.event_system = EventSystem()
        
        # Geometry database
        self.entities: Dict[str, GeometryEntity] = {}
        
        # Operation history for parametric updates
        self.operation_history = []
        
        # Tolerance settings
        self.linear_tolerance = 1e-6
        self.angular_tolerance = 1e-9
        
        # Performance settings
        self.tessellation_cache = {}
        self.max_cache_size = 1000
    
    async def initialize(self, app_context):
        """Initialize the geometry kernel"""
        self.app_context = app_context
        self.logger.info("Geometry kernel initialized")
    
    def create_point(self, x: float, y: float, z: float) -> str:
        """Create a point"""
        point = Point(x, y, z)
        self.entities[point.id] = point
        
        self.logger.debug(f"Created point: {point.id}")
        self.event_system.emit('geometry_created', {
            'entity_id': point.id,
            'type': 'point'
        })
        
        return point.id
    
    def create_line(self, start_point: List[float], end_point: List[float]) -> str:
        """Create a line"""
        line = Line(np.array(start_point), np.array(end_point))
        self.entities[line.id] = line
        
        self.logger.debug(f"Created line: {line.id}")
        self.event_system.emit('geometry_created', {
            'entity_id': line.id,
            'type': 'line'
        })
        
        return line.id
    
    def create_circle(self, center: List[float], radius: float, normal: List[float] = None) -> str:
        """Create a circle"""
        center_np = np.array(center)
        normal_np = np.array(normal) if normal else np.array([0, 0, 1])
        
        circle = Circle(center_np, radius, normal_np)
        self.entities[circle.id] = circle
        
        self.logger.debug(f"Created circle: {circle.id}")
        self.event_system.emit('geometry_created', {
            'entity_id': circle.id,
            'type': 'circle'
        })
        
        return circle.id
    
    def get_entity(self, entity_id: str) -> Optional[GeometryEntity]:
        """Get a geometry entity by ID"""
        return self.entities.get(entity_id)
    
    def update_entity(self, entity_id: str, **parameters) -> bool:
        """Update entity parameters"""
        entity = self.entities.get(entity_id)
        if not entity:
            return False
        
        entity.update_parameters(**parameters)
        
        self.event_system.emit('geometry_updated', {
            'entity_id': entity_id,
            'parameters': parameters
        })
        
        return True
    
    def delete_entity(self, entity_id: str) -> bool:
        """Delete a geometry entity"""
        if entity_id not in self.entities:
            return False
        
        del self.entities[entity_id]
        
        # Clear from tessellation cache
        if entity_id in self.tessellation_cache:
            del self.tessellation_cache[entity_id]
        
        self.event_system.emit('geometry_deleted', {
            'entity_id': entity_id
        })
        
        return True
    
    def get_tessellation(self, entity_id: str, tolerance: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get tessellation data for an entity"""
        entity = self.entities.get(entity_id)
        if not entity:
            return None
        
        # Check cache
        cache_key = f"{entity_id}_{tolerance}"
        if cache_key in self.tessellation_cache:
            return self.tessellation_cache[cache_key]
        
        # Generate tessellation
        tessellation = entity.get_tessellation(tolerance)
        
        # Cache result
        if len(self.tessellation_cache) < self.max_cache_size:
            self.tessellation_cache[cache_key] = tessellation
        
        return tessellation
    
    def get_bounding_box(self, entity_ids: List[str] = None) -> BoundingBox:
        """Get combined bounding box of entities"""
        if entity_ids is None:
            entity_ids = list(self.entities.keys())
        
        combined_bbox = BoundingBox()
        
        for entity_id in entity_ids:
            entity = self.entities.get(entity_id)
            if entity:
                entity_bbox = entity.get_world_bounding_box()
                combined_bbox = combined_bbox.union(entity_bbox)
        
        return combined_bbox
    
    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all entities"""
        validation_results = {}
        
        for entity_id, entity in self.entities.items():
            if not entity.validate():
                validation_results[entity_id] = entity.validation_errors
        
        return validation_results
    
    def clear_cache(self):
        """Clear tessellation cache"""
        self.tessellation_cache.clear()
        self.logger.debug("Tessellation cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get kernel statistics"""
        entity_counts = {}
        for entity in self.entities.values():
            entity_type = entity.geometry_type.value
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        return {
            'total_entities': len(self.entities),
            'entity_types': entity_counts,
            'cache_size': len(self.tessellation_cache),
            'linear_tolerance': self.linear_tolerance,
            'angular_tolerance': self.angular_tolerance
        }