"""
CAM Engine for OmniCAD

Provides comprehensive Computer-Aided Manufacturing capabilities including:
- Toolpath generation for 2D/3D machining
- Machine simulation and collision detection
- G-code generation and post-processing
- Tool management and feeds/speeds calculation
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import math

from ..utils.logger import OmniLogger
from ..utils.event_system import EventSystem
from ..geometry.kernel import GeometryEntity, Transform3D


class OperationType(Enum):
    """CAM operation types"""
    # 2D Operations
    CONTOUR = "contour"
    POCKET = "pocket"
    DRILL = "drill"
    FACE = "face"
    
    # 3D Operations
    ROUGH = "rough"
    FINISH = "finish"
    PENCIL = "pencil"
    
    # Advanced
    ADAPTIVE = "adaptive"
    TROCHOIDAL = "trochoidal"
    MULTIAXIS = "multiaxis"


class ToolType(Enum):
    """Tool types"""
    END_MILL = "end_mill"
    BALL_MILL = "ball_mill"
    DRILL = "drill"
    FACE_MILL = "face_mill"
    CHAMFER = "chamfer"
    TAP = "tap"


@dataclass
class Tool:
    """Cutting tool definition"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    tool_type: ToolType = ToolType.END_MILL
    diameter: float = 6.0  # mm
    length: float = 50.0
    flutes: int = 4
    helix_angle: float = 30.0
    material: str = "Carbide"
    coating: str = "TiAlN"
    
    # Cutting parameters
    max_rpm: int = 20000
    max_feed: float = 1000.0  # mm/min
    max_doc: float = 2.0      # depth of cut
    max_woc: float = 3.0      # width of cut


@dataclass
class CuttingParameters:
    """Cutting parameters for operations"""
    spindle_speed: int = 10000  # RPM
    feed_rate: float = 500.0    # mm/min
    plunge_rate: float = 200.0  # mm/min
    depth_per_pass: float = 1.0 # mm
    stepover: float = 2.0       # mm
    approach_distance: float = 2.0
    retract_distance: float = 2.0


@dataclass
class CAMSetup:
    """CAM setup/fixture definition"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    origin: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    work_coordinate_system: Transform3D = field(default_factory=Transform3D.identity)
    stock_geometry: Optional[Dict[str, Any]] = None
    fixtures: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Toolpath:
    """Generated toolpath data"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_id: str = ""
    tool_id: str = ""
    points: np.ndarray = field(default_factory=lambda: np.array([]))
    moves: List[Dict[str, Any]] = field(default_factory=list)
    cutting_time: float = 0.0
    rapid_time: float = 0.0
    total_length: float = 0.0


class ToolpathGenerator:
    """Generates toolpaths for various CAM operations"""
    
    def __init__(self):
        self.logger = OmniLogger("ToolpathGenerator")
    
    def generate_contour_2d(self, geometry: List[GeometryEntity], tool: Tool, params: CuttingParameters) -> Toolpath:
        """Generate 2D contour toolpath"""
        toolpath = Toolpath()
        moves = []
        
        # Simple contour following
        for entity in geometry:
            if hasattr(entity, 'get_tessellation'):
                tess = entity.get_tessellation(0.1)
                vertices = tess['vertices']
                
                # Offset by tool radius
                offset_distance = tool.diameter / 2
                offset_vertices = self._offset_curve_2d(vertices, offset_distance)
                
                # Create moves
                for i, point in enumerate(offset_vertices):
                    if i == 0:
                        # Rapid to approach point
                        approach_point = point + np.array([0, 0, params.approach_distance])
                        moves.append({
                            'type': 'rapid',
                            'position': approach_point,
                            'feed_rate': 0
                        })
                        # Plunge to cutting depth
                        moves.append({
                            'type': 'linear',
                            'position': point,
                            'feed_rate': params.plunge_rate
                        })
                    else:
                        # Linear cutting move
                        moves.append({
                            'type': 'linear',
                            'position': point,
                            'feed_rate': params.feed_rate
                        })
        
        # Retract
        if moves:
            last_point = moves[-1]['position']
            retract_point = last_point + np.array([0, 0, params.retract_distance])
            moves.append({
                'type': 'rapid',
                'position': retract_point,
                'feed_rate': 0
            })
        
        toolpath.moves = moves
        toolpath.total_length = self._calculate_toolpath_length(moves)
        toolpath.cutting_time = self._calculate_cutting_time(moves, params)
        
        return toolpath
    
    def generate_pocket_2d(self, boundary: List[GeometryEntity], tool: Tool, params: CuttingParameters) -> Toolpath:
        """Generate 2D pocket clearing toolpath"""
        toolpath = Toolpath()
        moves = []
        
        # Get boundary tessellation
        boundary_points = []
        for entity in boundary:
            if hasattr(entity, 'get_tessellation'):
                tess = entity.get_tessellation(0.1)
                boundary_points.extend(tess['vertices'])
        
        if not boundary_points:
            return toolpath
        
        boundary_array = np.array(boundary_points)
        
        # Calculate bounding box
        min_x, max_x = np.min(boundary_array[:, 0]), np.max(boundary_array[:, 0])
        min_y, max_y = np.min(boundary_array[:, 1]), np.max(boundary_array[:, 1])
        
        # Generate parallel passes
        tool_radius = tool.diameter / 2
        stepover = params.stepover
        
        y = min_y + tool_radius
        direction = 1
        
        while y <= max_y - tool_radius:
            # Create horizontal pass
            start_x = min_x + tool_radius if direction > 0 else max_x - tool_radius
            end_x = max_x - tool_radius if direction > 0 else min_x + tool_radius
            
            start_point = np.array([start_x, y, 0])
            end_point = np.array([end_x, y, 0])
            
            # Add rapid to start
            if not moves:
                approach_point = start_point + np.array([0, 0, params.approach_distance])
                moves.append({
                    'type': 'rapid',
                    'position': approach_point,
                    'feed_rate': 0
                })
                moves.append({
                    'type': 'linear',
                    'position': start_point,
                    'feed_rate': params.plunge_rate
                })
            else:
                moves.append({
                    'type': 'linear',
                    'position': start_point,
                    'feed_rate': params.feed_rate
                })
            
            # Add cutting move
            moves.append({
                'type': 'linear',
                'position': end_point,
                'feed_rate': params.feed_rate
            })
            
            y += stepover
            direction *= -1
        
        # Final retract
        if moves:
            last_point = moves[-1]['position']
            retract_point = last_point + np.array([0, 0, params.retract_distance])
            moves.append({
                'type': 'rapid',
                'position': retract_point,
                'feed_rate': 0
            })
        
        toolpath.moves = moves
        toolpath.total_length = self._calculate_toolpath_length(moves)
        toolpath.cutting_time = self._calculate_cutting_time(moves, params)
        
        return toolpath
    
    def generate_drill_holes(self, hole_positions: List[np.ndarray], tool: Tool, params: CuttingParameters) -> Toolpath:
        """Generate drill hole toolpath"""
        toolpath = Toolpath()
        moves = []
        
        for i, position in enumerate(hole_positions):
            # Rapid to approach height
            approach_point = position + np.array([0, 0, params.approach_distance])
            moves.append({
                'type': 'rapid',
                'position': approach_point,
                'feed_rate': 0
            })
            
            # Drill down
            bottom_point = position - np.array([0, 0, params.depth_per_pass])
            moves.append({
                'type': 'linear',
                'position': bottom_point,
                'feed_rate': params.plunge_rate
            })
            
            # Retract
            moves.append({
                'type': 'rapid',
                'position': approach_point,
                'feed_rate': 0
            })
        
        toolpath.moves = moves
        toolpath.total_length = self._calculate_toolpath_length(moves)
        toolpath.cutting_time = self._calculate_cutting_time(moves, params)
        
        return toolpath
    
    def _offset_curve_2d(self, vertices: np.ndarray, offset_distance: float) -> np.ndarray:
        """Offset a 2D curve by specified distance"""
        if len(vertices) < 2:
            return vertices
        
        offset_vertices = []
        
        for i in range(len(vertices)):
            # Get current and next vertex
            current = vertices[i][:2]  # Take only X,Y
            next_vertex = vertices[(i + 1) % len(vertices)][:2]
            
            # Calculate edge direction
            edge_dir = next_vertex - current
            if np.linalg.norm(edge_dir) > 1e-12:
                edge_dir = edge_dir / np.linalg.norm(edge_dir)
                
                # Calculate normal (perpendicular)
                normal = np.array([-edge_dir[1], edge_dir[0]])
                
                # Offset point
                offset_point = current + normal * offset_distance
                offset_vertices.append(np.array([offset_point[0], offset_point[1], current[2] if len(current) > 2 else 0]))
        
        return np.array(offset_vertices) if offset_vertices else vertices
    
    def _calculate_toolpath_length(self, moves: List[Dict[str, Any]]) -> float:
        """Calculate total toolpath length"""
        total_length = 0.0
        previous_pos = None
        
        for move in moves:
            current_pos = move['position']
            if previous_pos is not None:
                distance = np.linalg.norm(current_pos - previous_pos)
                total_length += distance
            previous_pos = current_pos
        
        return total_length
    
    def _calculate_cutting_time(self, moves: List[Dict[str, Any]], params: CuttingParameters) -> float:
        """Calculate estimated cutting time"""
        cutting_time = 0.0
        previous_pos = None
        
        for move in moves:
            current_pos = move['position']
            feed_rate = move.get('feed_rate', params.feed_rate)
            
            if previous_pos is not None and feed_rate > 0:
                distance = np.linalg.norm(current_pos - previous_pos)
                time = distance / feed_rate * 60  # Convert mm/min to seconds
                cutting_time += time
            
            previous_pos = current_pos
        
        return cutting_time


class GCodeGenerator:
    """Generates G-code from toolpaths"""
    
    def __init__(self):
        self.logger = OmniLogger("GCodeGenerator")
    
    def generate_gcode(self, toolpath: Toolpath, setup: CAMSetup, tool: Tool, params: CuttingParameters) -> str:
        """Generate G-code from toolpath"""
        gcode_lines = []
        
        # Header
        gcode_lines.extend([
            "; Generated by OmniCAD CAM",
            f"; Date: {self._get_current_date()}",
            f"; Tool: {tool.name} ({tool.diameter}mm)",
            "",
            "G21 ; Set units to millimeters",
            "G90 ; Use absolute positioning",
            "G17 ; Select XY plane",
            "G40 ; Cancel cutter compensation",
            "G49 ; Cancel tool length compensation",
            "",
            f"M6 T{1} ; Tool change",
            f"S{params.spindle_speed} M3 ; Start spindle",
            "G0 Z25 ; Rapid to safe height",
            ""
        ])
        
        # Process moves
        for i, move in enumerate(toolpath.moves):
            pos = move['position']
            feed_rate = move.get('feed_rate', params.feed_rate)
            
            if move['type'] == 'rapid':
                gcode_lines.append(f"G0 X{pos[0]:.3f} Y{pos[1]:.3f} Z{pos[2]:.3f}")
            elif move['type'] == 'linear':
                if feed_rate > 0:
                    gcode_lines.append(f"G1 X{pos[0]:.3f} Y{pos[1]:.3f} Z{pos[2]:.3f} F{feed_rate:.0f}")
                else:
                    gcode_lines.append(f"G1 X{pos[0]:.3f} Y{pos[1]:.3f} Z{pos[2]:.3f}")
            elif move['type'] == 'arc_cw':
                # Clockwise arc (would need I,J parameters)
                gcode_lines.append(f"G2 X{pos[0]:.3f} Y{pos[1]:.3f} Z{pos[2]:.3f}")
            elif move['type'] == 'arc_ccw':
                # Counter-clockwise arc
                gcode_lines.append(f"G3 X{pos[0]:.3f} Y{pos[1]:.3f} Z{pos[2]:.3f}")
        
        # Footer
        gcode_lines.extend([
            "",
            "G0 Z25 ; Rapid to safe height",
            "M5 ; Stop spindle",
            "M30 ; Program end"
        ])
        
        return "\n".join(gcode_lines)
    
    def _get_current_date(self) -> str:
        """Get current date string"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class CAMEngine:
    """Main CAM engine for OmniCAD"""
    
    def __init__(self):
        self.logger = OmniLogger("CAMEngine")
        self.event_system = EventSystem()
        
        # CAM data
        self.setups: Dict[str, CAMSetup] = {}
        self.tools: Dict[str, Tool] = {}
        self.operations: Dict[str, Dict[str, Any]] = {}
        self.toolpaths: Dict[str, Toolpath] = {}
        
        # Generators
        self.toolpath_generator = ToolpathGenerator()
        self.gcode_generator = GCodeGenerator()
        
        # Initialize default tools
        self._create_default_tools()
    
    async def initialize(self, app_context):
        """Initialize CAM engine"""
        self.app_context = app_context
        self.logger.info("CAM Engine initialized")
    
    def _create_default_tools(self):
        """Create default tool library"""
        tools = [
            Tool(name="6mm End Mill", diameter=6.0, tool_type=ToolType.END_MILL),
            Tool(name="10mm End Mill", diameter=10.0, tool_type=ToolType.END_MILL),
            Tool(name="3mm Ball Mill", diameter=3.0, tool_type=ToolType.BALL_MILL),
            Tool(name="6mm Drill", diameter=6.0, tool_type=ToolType.DRILL),
            Tool(name="50mm Face Mill", diameter=50.0, tool_type=ToolType.FACE_MILL)
        ]
        
        for tool in tools:
            self.tools[tool.id] = tool
    
    def create_setup(self, name: str, origin: np.ndarray = None) -> str:
        """Create a new CAM setup"""
        setup = CAMSetup(
            name=name,
            origin=origin or np.array([0, 0, 0])
        )
        
        self.setups[setup.id] = setup
        
        self.event_system.emit('cam_setup_created', {
            'setup_id': setup.id,
            'name': name
        })
        
        return setup.id
    
    def create_operation(self, operation_type: OperationType, name: str, setup_id: str, tool_id: str, geometry: List[GeometryEntity]) -> str:
        """Create a CAM operation"""
        operation_id = str(uuid.uuid4())
        
        operation = {
            'id': operation_id,
            'name': name,
            'type': operation_type,
            'setup_id': setup_id,
            'tool_id': tool_id,
            'geometry': geometry,
            'parameters': CuttingParameters(),
            'toolpath_id': None
        }
        
        self.operations[operation_id] = operation
        
        self.event_system.emit('cam_operation_created', {
            'operation_id': operation_id,
            'type': operation_type.value
        })
        
        return operation_id
    
    def generate_toolpath(self, operation_id: str) -> str:
        """Generate toolpath for operation"""
        operation = self.operations.get(operation_id)
        if not operation:
            raise ValueError(f"Operation not found: {operation_id}")
        
        tool = self.tools.get(operation['tool_id'])
        if not tool:
            raise ValueError(f"Tool not found: {operation['tool_id']}")
        
        params = operation['parameters']
        geometry = operation['geometry']
        op_type = operation['type']
        
        # Generate toolpath based on operation type
        if op_type == OperationType.CONTOUR:
            toolpath = self.toolpath_generator.generate_contour_2d(geometry, tool, params)
        elif op_type == OperationType.POCKET:
            toolpath = self.toolpath_generator.generate_pocket_2d(geometry, tool, params)
        elif op_type == OperationType.DRILL:
            # Extract hole positions from geometry
            hole_positions = [np.array([0, 0, 0])]  # Placeholder
            toolpath = self.toolpath_generator.generate_drill_holes(hole_positions, tool, params)
        else:
            raise ValueError(f"Unsupported operation type: {op_type}")
        
        toolpath.operation_id = operation_id
        toolpath.tool_id = operation['tool_id']
        
        self.toolpaths[toolpath.id] = toolpath
        operation['toolpath_id'] = toolpath.id
        
        self.event_system.emit('toolpath_generated', {
            'operation_id': operation_id,
            'toolpath_id': toolpath.id
        })
        
        return toolpath.id
    
    def generate_gcode(self, operation_id: str) -> str:
        """Generate G-code for operation"""
        operation = self.operations.get(operation_id)
        if not operation:
            raise ValueError(f"Operation not found: {operation_id}")
        
        toolpath_id = operation.get('toolpath_id')
        if not toolpath_id:
            raise ValueError(f"No toolpath generated for operation: {operation_id}")
        
        toolpath = self.toolpaths.get(toolpath_id)
        setup = self.setups.get(operation['setup_id'])
        tool = self.tools.get(operation['tool_id'])
        params = operation['parameters']
        
        if not all([toolpath, setup, tool]):
            raise ValueError("Missing required data for G-code generation")
        
        gcode = self.gcode_generator.generate_gcode(toolpath, setup, tool, params)
        
        self.event_system.emit('gcode_generated', {
            'operation_id': operation_id,
            'gcode_length': len(gcode)
        })
        
        return gcode
    
    def get_operation_statistics(self, operation_id: str) -> Dict[str, Any]:
        """Get operation statistics"""
        operation = self.operations.get(operation_id)
        if not operation:
            return {}
        
        toolpath_id = operation.get('toolpath_id')
        toolpath = self.toolpaths.get(toolpath_id) if toolpath_id else None
        
        stats = {
            'operation_name': operation['name'],
            'operation_type': operation['type'].value,
            'tool_name': self.tools.get(operation['tool_id'], Tool()).name,
            'has_toolpath': toolpath is not None
        }
        
        if toolpath:
            stats.update({
                'total_length': toolpath.total_length,
                'cutting_time': toolpath.cutting_time,
                'rapid_time': toolpath.rapid_time,
                'move_count': len(toolpath.moves)
            })
        
        return stats
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get overall CAM statistics"""
        return {
            'total_setups': len(self.setups),
            'total_tools': len(self.tools),
            'total_operations': len(self.operations),
            'total_toolpaths': len(self.toolpaths),
            'completed_operations': sum(1 for op in self.operations.values() 
                                      if op.get('toolpath_id') is not None)
        }
    
    def shutdown(self):
        """Shutdown CAM engine"""
        self.logger.info("CAM Engine shutdown")