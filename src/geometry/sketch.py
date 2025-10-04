"""
2D Sketching Engine for OmniCAD

Provides parametric 2D sketching capabilities with constraint solving.
This forms the foundation for feature-based modeling.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import math

from ..utils.logger import OmniLogger
from ..utils.event_system import EventSystem
from .kernel import GeometryEntity, GeometryType, Transform3D, Point, Line, Circle


class ConstraintType(Enum):
    """Types of geometric constraints"""
    # Distance constraints
    DISTANCE = "distance"
    HORIZONTAL_DISTANCE = "horizontal_distance"
    VERTICAL_DISTANCE = "vertical_distance"
    
    # Angle constraints
    ANGLE = "angle"
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    
    # Position constraints
    COINCIDENT = "coincident"
    CONCENTRIC = "concentric"
    MIDPOINT = "midpoint"
    
    # Geometric constraints
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    TANGENT = "tangent"
    EQUAL_RADIUS = "equal_radius"
    EQUAL_LENGTH = "equal_length"
    
    # Advanced constraints
    SYMMETRIC = "symmetric"
    PATTERN = "pattern"


@dataclass
class SketchConstraint:
    """Represents a geometric constraint in a sketch"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    constraint_type: ConstraintType = ConstraintType.DISTANCE
    entities: List[str] = field(default_factory=list)  # Entity IDs
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_satisfied: bool = False
    error: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'type': self.constraint_type.value,
            'entities': self.entities,
            'parameters': self.parameters,
            'is_satisfied': self.is_satisfied,
            'error': self.error
        }


class SketchGeometry(GeometryEntity):
    """Base class for sketch geometry entities"""
    
    def __init__(self, geometry_type: GeometryType, parameters: Dict[str, Any] = None):
        super().__init__(geometry_type, parameters)
        self.is_construction = False
        self.is_fixed = False
        self.sketch_id: Optional[str] = None
    
    def set_construction(self, is_construction: bool):
        """Set construction mode"""
        self.is_construction = is_construction
        if is_construction:
            self.visual_properties['color'] = [0.5, 0.5, 0.5, 0.7]
        else:
            self.visual_properties['color'] = [0.2, 0.2, 0.8, 1.0]
    
    def set_fixed(self, is_fixed: bool):
        """Set fixed constraint"""
        self.is_fixed = is_fixed


class SketchPoint(SketchGeometry):
    """2D Point in sketch"""
    
    def __init__(self, x: float = 0, y: float = 0):
        super().__init__(GeometryType.POINT, {
            'x': x, 'y': y
        })
        self.regenerate()
    
    def _compute_geometry(self):
        """Compute point geometry"""
        self.point_2d = np.array([
            self.parameters['x'],
            self.parameters['y']
        ])
        # Z coordinate is 0 for sketch
        self.point_3d = np.array([
            self.parameters['x'],
            self.parameters['y'],
            0.0
        ])
    
    def _update_bounding_box(self):
        """Update bounding box"""
        from .kernel import BoundingBox
        self.bounding_box = BoundingBox(self.point_3d.copy(), self.point_3d.copy())


class SketchLine(SketchGeometry):
    """2D Line in sketch"""
    
    def __init__(self, start_point_id: str, end_point_id: str):
        super().__init__(GeometryType.LINE, {
            'start_point_id': start_point_id,
            'end_point_id': end_point_id
        })
        self.start_point_id = start_point_id
        self.end_point_id = end_point_id
    
    def _compute_geometry(self):
        """Compute line geometry (requires point references)"""
        # This will be computed when the sketch is solved
        pass
    
    def compute_from_points(self, start_point: SketchPoint, end_point: SketchPoint):
        """Compute geometry from point objects"""
        self.start_point_2d = start_point.point_2d
        self.end_point_2d = end_point.point_2d
        self.direction_2d = self.end_point_2d - self.start_point_2d
        self.length = np.linalg.norm(self.direction_2d)
        
        if self.length > 0:
            self.unit_direction_2d = self.direction_2d / self.length
        else:
            self.unit_direction_2d = np.array([1, 0])
        
        # Update 3D representation
        self.start_point_3d = np.append(self.start_point_2d, 0.0)
        self.end_point_3d = np.append(self.end_point_2d, 0.0)


class SketchCircle(SketchGeometry):
    """2D Circle in sketch"""
    
    def __init__(self, center_point_id: str, radius: float):
        super().__init__(GeometryType.CIRCLE, {
            'center_point_id': center_point_id,
            'radius': radius
        })
        self.center_point_id = center_point_id
    
    def _compute_geometry(self):
        """Compute circle geometry (requires point reference)"""
        pass
    
    def compute_from_center(self, center_point: SketchPoint):
        """Compute geometry from center point"""
        self.center_2d = center_point.point_2d
        self.radius = self.parameters['radius']
        
        # Update 3D representation
        self.center_3d = np.append(self.center_2d, 0.0)


class ConstraintSolver:
    """
    Parametric constraint solver for 2D sketches.
    
    Uses a simple iterative solver with gradient descent.
    For production use, this would be replaced with a more robust solver
    like SolveSpace or FreeCAD's constraint solver.
    """
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 100):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.logger = OmniLogger("ConstraintSolver")
    
    def solve(self, sketch: 'Sketch') -> bool:
        """Solve all constraints in the sketch"""
        try:
            # Collect all variables (point coordinates)
            variables = self._collect_variables(sketch)
            
            # Solve iteratively
            for iteration in range(self.max_iterations):
                # Evaluate constraints
                residuals = self._evaluate_constraints(sketch, variables)
                
                # Check convergence
                max_error = max(abs(r) for r in residuals) if residuals else 0
                if max_error < self.tolerance:
                    self.logger.debug(f"Constraints solved in {iteration} iterations")
                    self._update_sketch_geometry(sketch, variables)
                    return True
                
                # Update variables using gradient descent
                variables = self._gradient_descent_step(sketch, variables, residuals)
            
            self.logger.warning(f"Constraint solver did not converge after {self.max_iterations} iterations")
            self._update_sketch_geometry(sketch, variables)
            return False
            
        except Exception as e:
            self.logger.error(f"Constraint solver error: {str(e)}")
            return False
    
    def _collect_variables(self, sketch: 'Sketch') -> np.ndarray:
        """Collect all free variables (non-fixed point coordinates)"""
        variables = []
        variable_map = {}
        index = 0
        
        for entity_id, entity in sketch.entities.items():
            if isinstance(entity, SketchPoint) and not entity.is_fixed:
                variables.extend([entity.parameters['x'], entity.parameters['y']])
                variable_map[entity_id] = (index, index + 1)
                index += 2
        
        sketch._variable_map = variable_map
        return np.array(variables)
    
    def _evaluate_constraints(self, sketch: 'Sketch', variables: np.ndarray) -> List[float]:
        """Evaluate all constraints and return residuals"""
        residuals = []
        
        # Update entity positions from variables
        self._update_entity_positions(sketch, variables)
        
        for constraint in sketch.constraints.values():
            residual = self._evaluate_constraint(sketch, constraint)
            residuals.append(residual)
            constraint.error = abs(residual)
            constraint.is_satisfied = abs(residual) < self.tolerance
        
        return residuals
    
    def _evaluate_constraint(self, sketch: 'Sketch', constraint: SketchConstraint) -> float:
        """Evaluate a single constraint"""
        try:
            if constraint.constraint_type == ConstraintType.DISTANCE:
                return self._evaluate_distance_constraint(sketch, constraint)
            elif constraint.constraint_type == ConstraintType.COINCIDENT:
                return self._evaluate_coincident_constraint(sketch, constraint)
            elif constraint.constraint_type == ConstraintType.HORIZONTAL:
                return self._evaluate_horizontal_constraint(sketch, constraint)
            elif constraint.constraint_type == ConstraintType.VERTICAL:
                return self._evaluate_vertical_constraint(sketch, constraint)
            elif constraint.constraint_type == ConstraintType.PARALLEL:
                return self._evaluate_parallel_constraint(sketch, constraint)
            elif constraint.constraint_type == ConstraintType.PERPENDICULAR:
                return self._evaluate_perpendicular_constraint(sketch, constraint)
            else:
                return 0.0
        except Exception as e:
            self.logger.error(f"Error evaluating constraint {constraint.id}: {str(e)}")
            return 0.0
    
    def _evaluate_distance_constraint(self, sketch: 'Sketch', constraint: SketchConstraint) -> float:
        """Evaluate distance constraint between two points"""
        if len(constraint.entities) != 2:
            return 0.0
        
        entity1 = sketch.entities.get(constraint.entities[0])
        entity2 = sketch.entities.get(constraint.entities[1])
        
        if not (isinstance(entity1, SketchPoint) and isinstance(entity2, SketchPoint)):
            return 0.0
        
        target_distance = constraint.parameters.get('distance', 0.0)
        actual_distance = np.linalg.norm(entity1.point_2d - entity2.point_2d)
        
        return actual_distance - target_distance
    
    def _evaluate_coincident_constraint(self, sketch: 'Sketch', constraint: SketchConstraint) -> float:
        """Evaluate coincident constraint between two points"""
        if len(constraint.entities) != 2:
            return 0.0
        
        entity1 = sketch.entities.get(constraint.entities[0])
        entity2 = sketch.entities.get(constraint.entities[1])
        
        if not (isinstance(entity1, SketchPoint) and isinstance(entity2, SketchPoint)):
            return 0.0
        
        distance = np.linalg.norm(entity1.point_2d - entity2.point_2d)
        return distance
    
    def _evaluate_horizontal_constraint(self, sketch: 'Sketch', constraint: SketchConstraint) -> float:
        """Evaluate horizontal constraint on a line"""
        if len(constraint.entities) != 1:
            return 0.0
        
        entity = sketch.entities.get(constraint.entities[0])
        if not isinstance(entity, SketchLine):
            return 0.0
        
        # Get start and end points
        start_point = sketch.entities.get(entity.start_point_id)
        end_point = sketch.entities.get(entity.end_point_id)
        
        if not (isinstance(start_point, SketchPoint) and isinstance(end_point, SketchPoint)):
            return 0.0
        
        # Return Y difference (should be 0 for horizontal line)
        return end_point.point_2d[1] - start_point.point_2d[1]
    
    def _evaluate_vertical_constraint(self, sketch: 'Sketch', constraint: SketchConstraint) -> float:
        """Evaluate vertical constraint on a line"""
        if len(constraint.entities) != 1:
            return 0.0
        
        entity = sketch.entities.get(constraint.entities[0])
        if not isinstance(entity, SketchLine):
            return 0.0
        
        # Get start and end points
        start_point = sketch.entities.get(entity.start_point_id)
        end_point = sketch.entities.get(entity.end_point_id)
        
        if not (isinstance(start_point, SketchPoint) and isinstance(end_point, SketchPoint)):
            return 0.0
        
        # Return X difference (should be 0 for vertical line)
        return end_point.point_2d[0] - start_point.point_2d[0]
    
    def _evaluate_parallel_constraint(self, sketch: 'Sketch', constraint: SketchConstraint) -> float:
        """Evaluate parallel constraint between two lines"""
        if len(constraint.entities) != 2:
            return 0.0
        
        line1 = sketch.entities.get(constraint.entities[0])
        line2 = sketch.entities.get(constraint.entities[1])
        
        if not (isinstance(line1, SketchLine) and isinstance(line2, SketchLine)):
            return 0.0
        
        # Get direction vectors
        start1 = sketch.entities.get(line1.start_point_id)
        end1 = sketch.entities.get(line1.end_point_id)
        start2 = sketch.entities.get(line2.start_point_id)
        end2 = sketch.entities.get(line2.end_point_id)
        
        if not all(isinstance(p, SketchPoint) for p in [start1, end1, start2, end2]):
            return 0.0
        
        dir1 = end1.point_2d - start1.point_2d
        dir2 = end2.point_2d - start2.point_2d
        
        # Normalize directions
        if np.linalg.norm(dir1) > 0:
            dir1 = dir1 / np.linalg.norm(dir1)
        if np.linalg.norm(dir2) > 0:
            dir2 = dir2 / np.linalg.norm(dir2)
        
        # Cross product should be 0 for parallel lines
        cross_product = dir1[0] * dir2[1] - dir1[1] * dir2[0]
        return cross_product
    
    def _evaluate_perpendicular_constraint(self, sketch: 'Sketch', constraint: SketchConstraint) -> float:
        """Evaluate perpendicular constraint between two lines"""
        if len(constraint.entities) != 2:
            return 0.0
        
        line1 = sketch.entities.get(constraint.entities[0])
        line2 = sketch.entities.get(constraint.entities[1])
        
        if not (isinstance(line1, SketchLine) and isinstance(line2, SketchLine)):
            return 0.0
        
        # Get direction vectors
        start1 = sketch.entities.get(line1.start_point_id)
        end1 = sketch.entities.get(line1.end_point_id)
        start2 = sketch.entities.get(line2.start_point_id)
        end2 = sketch.entities.get(line2.end_point_id)
        
        if not all(isinstance(p, SketchPoint) for p in [start1, end1, start2, end2]):
            return 0.0
        
        dir1 = end1.point_2d - start1.point_2d
        dir2 = end2.point_2d - start2.point_2d
        
        # Normalize directions
        if np.linalg.norm(dir1) > 0:
            dir1 = dir1 / np.linalg.norm(dir1)
        if np.linalg.norm(dir2) > 0:
            dir2 = dir2 / np.linalg.norm(dir2)
        
        # Dot product should be 0 for perpendicular lines
        dot_product = np.dot(dir1, dir2)
        return dot_product
    
    def _update_entity_positions(self, sketch: 'Sketch', variables: np.ndarray):
        """Update entity positions from variables"""
        for entity_id, (x_idx, y_idx) in sketch._variable_map.items():
            entity = sketch.entities.get(entity_id)
            if isinstance(entity, SketchPoint):
                entity.parameters['x'] = variables[x_idx]
                entity.parameters['y'] = variables[y_idx]
                entity._compute_geometry()
    
    def _gradient_descent_step(self, sketch: 'Sketch', variables: np.ndarray, residuals: List[float]) -> np.ndarray:
        """Perform one gradient descent step"""
        learning_rate = 0.1
        gradient = np.zeros_like(variables)
        
        # Compute numerical gradient
        epsilon = 1e-8
        for i in range(len(variables)):
            variables_plus = variables.copy()
            variables_plus[i] += epsilon
            
            residuals_plus = self._evaluate_constraints(sketch, variables_plus)
            
            # Compute partial derivative
            for j, (res_plus, res) in enumerate(zip(residuals_plus, residuals)):
                gradient[i] += (res_plus - res) / epsilon * res
        
        # Update variables
        return variables - learning_rate * gradient
    
    def _update_sketch_geometry(self, sketch: 'Sketch', variables: np.ndarray):
        """Update all sketch geometry after solving"""
        self._update_entity_positions(sketch, variables)
        
        # Update line geometry
        for entity in sketch.entities.values():
            if isinstance(entity, SketchLine):
                start_point = sketch.entities.get(entity.start_point_id)
                end_point = sketch.entities.get(entity.end_point_id)
                if isinstance(start_point, SketchPoint) and isinstance(end_point, SketchPoint):
                    entity.compute_from_points(start_point, end_point)
            elif isinstance(entity, SketchCircle):
                center_point = sketch.entities.get(entity.center_point_id)
                if isinstance(center_point, SketchPoint):
                    entity.compute_from_center(center_point)


class Sketch:
    """
    2D Parametric Sketch
    
    Contains geometric entities and constraints that define a 2D profile.
    This can be used as the basis for 3D features like extrusions.
    """
    
    def __init__(self, name: str = "Sketch"):
        self.id = str(uuid.uuid4())
        self.name = name
        
        # Geometry
        self.entities: Dict[str, SketchGeometry] = {}
        self.constraints: Dict[str, SketchConstraint] = {}
        
        # Sketch plane
        self.plane_origin = np.array([0, 0, 0])
        self.plane_normal = np.array([0, 0, 1])
        self.plane_x_axis = np.array([1, 0, 0])
        self.plane_y_axis = np.array([0, 1, 0])
        
        # State
        self.is_solved = False
        self.is_fully_constrained = False
        self.degrees_of_freedom = 0
        
        # Solver
        self.solver = ConstraintSolver()
        
        # Internal variables for solving
        self._variable_map = {}
        
        self.logger = OmniLogger("Sketch")
        self.event_system = EventSystem()
    
    def add_point(self, x: float, y: float) -> str:
        """Add a point to the sketch"""
        point = SketchPoint(x, y)
        point.sketch_id = self.id
        self.entities[point.id] = point
        
        self.logger.debug(f"Added point to sketch: {point.id}")
        self.event_system.emit('sketch_entity_added', {
            'sketch_id': self.id,
            'entity_id': point.id,
            'entity_type': 'point'
        })
        
        return point.id
    
    def add_line(self, start_point_id: str, end_point_id: str) -> str:
        """Add a line to the sketch"""
        line = SketchLine(start_point_id, end_point_id)
        line.sketch_id = self.id
        self.entities[line.id] = line
        
        self.logger.debug(f"Added line to sketch: {line.id}")
        self.event_system.emit('sketch_entity_added', {
            'sketch_id': self.id,
            'entity_id': line.id,
            'entity_type': 'line'
        })
        
        return line.id
    
    def add_circle(self, center_point_id: str, radius: float) -> str:
        """Add a circle to the sketch"""
        circle = SketchCircle(center_point_id, radius)
        circle.sketch_id = self.id
        self.entities[circle.id] = circle
        
        self.logger.debug(f"Added circle to sketch: {circle.id}")
        self.event_system.emit('sketch_entity_added', {
            'sketch_id': self.id,
            'entity_id': circle.id,
            'entity_type': 'circle'
        })
        
        return circle.id
    
    def add_constraint(self, constraint_type: ConstraintType, entities: List[str], **parameters) -> str:
        """Add a constraint to the sketch"""
        constraint = SketchConstraint(
            constraint_type=constraint_type,
            entities=entities,
            parameters=parameters
        )
        
        self.constraints[constraint.id] = constraint
        
        self.logger.debug(f"Added constraint to sketch: {constraint.id}")
        self.event_system.emit('sketch_constraint_added', {
            'sketch_id': self.id,
            'constraint_id': constraint.id,
            'constraint_type': constraint_type.value
        })
        
        return constraint.id
    
    def solve(self) -> bool:
        """Solve all constraints in the sketch"""
        self.logger.debug(f"Solving sketch: {self.id}")
        
        self.is_solved = self.solver.solve(self)
        
        if self.is_solved:
            self.logger.debug(f"Sketch solved successfully: {self.id}")
        else:
            self.logger.warning(f"Sketch solve failed: {self.id}")
        
        self.event_system.emit('sketch_solved', {
            'sketch_id': self.id,
            'success': self.is_solved
        })
        
        return self.is_solved
    
    def get_profile_curves(self) -> List[SketchGeometry]:
        """Get all non-construction curves for creating 3D features"""
        curves = []
        for entity in self.entities.values():
            if not entity.is_construction and not isinstance(entity, SketchPoint):
                curves.append(entity)
        return curves
    
    def get_bounding_box_2d(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get 2D bounding box of sketch"""
        if not self.entities:
            return np.zeros(2), np.zeros(2)
        
        min_pt = np.array([float('inf'), float('inf')])
        max_pt = np.array([float('-inf'), float('-inf')])
        
        for entity in self.entities.values():
            if isinstance(entity, SketchPoint):
                min_pt = np.minimum(min_pt, entity.point_2d)
                max_pt = np.maximum(max_pt, entity.point_2d)
        
        return min_pt, max_pt
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sketch to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'entities': {eid: entity.to_dict() for eid, entity in self.entities.items()},
            'constraints': {cid: constraint.to_dict() for cid, constraint in self.constraints.items()},
            'plane_origin': self.plane_origin.tolist(),
            'plane_normal': self.plane_normal.tolist(),
            'plane_x_axis': self.plane_x_axis.tolist(),
            'plane_y_axis': self.plane_y_axis.tolist(),
            'is_solved': self.is_solved,
            'is_fully_constrained': self.is_fully_constrained
        }