"""
Assembly Management System for OmniCAD

Provides hierarchical assembly modeling with constraint-based mates.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

from ..utils.logger import OmniLogger
from ..utils.event_system import EventSystem
from ..geometry.kernel import Transform3D, BoundingBox


class MateType(Enum):
    """Assembly mate types"""
    COINCIDENT = "coincident"
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    CONCENTRIC = "concentric"
    DISTANCE = "distance"
    ANGLE = "angle"


class ComponentState(Enum):
    """Component states"""
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    HIDDEN = "hidden"


@dataclass
class ComponentInstance:
    """Assembly component instance"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    source_path: str = ""
    transform: Transform3D = field(default_factory=Transform3D.identity)
    state: ComponentState = ComponentState.RESOLVED
    is_fixed: bool = False


@dataclass
class AssemblyMate:
    """Assembly mate constraint"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    mate_type: MateType = MateType.COINCIDENT
    component_ids: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_satisfied: bool = False


class AssemblyManager:
    """Main assembly manager"""
    
    def __init__(self):
        self.logger = OmniLogger("AssemblyManager")
        self.components: Dict[str, ComponentInstance] = {}
        self.mates: Dict[str, AssemblyMate] = {}
        self.event_system = EventSystem()
    
    async def initialize(self, app_context):
        """Initialize assembly manager"""
        self.app_context = app_context
        self.logger.info("Assembly Manager initialized")
    
    def add_component(self, name: str, source_path: str, transform: Transform3D = None) -> str:
        """Add component to assembly"""
        component = ComponentInstance(
            name=name,
            source_path=source_path,
            transform=transform or Transform3D.identity()
        )
        
        self.components[component.id] = component
        
        self.event_system.emit('component_added', {
            'component_id': component.id,
            'name': name
        })
        
        return component.id
    
    def add_mate(self, name: str, mate_type: MateType, component_ids: List[str], **params) -> str:
        """Add mate constraint"""
        mate = AssemblyMate(
            name=name,
            mate_type=mate_type,
            component_ids=component_ids,
            parameters=params
        )
        
        self.mates[mate.id] = mate
        
        self.event_system.emit('mate_added', {
            'mate_id': mate.id,
            'mate_type': mate_type.value
        })
        
        return mate.id
    
    def solve_assembly(self) -> bool:
        """Solve assembly constraints"""
        self.logger.debug("Solving assembly constraints")
        
        # Simple constraint satisfaction
        for mate in self.mates.values():
            if mate.mate_type == MateType.COINCIDENT:
                self._solve_coincident_mate(mate)
            elif mate.mate_type == MateType.DISTANCE:
                self._solve_distance_mate(mate)
        
        return True
    
    def _solve_coincident_mate(self, mate: AssemblyMate):
        """Solve coincident mate"""
        if len(mate.component_ids) >= 2:
            comp1 = self.components.get(mate.component_ids[0])
            comp2 = self.components.get(mate.component_ids[1])
            
            if comp1 and comp2 and not comp1.is_fixed and not comp2.is_fixed:
                # Simple positioning - move comp2 to comp1's position
                comp2.transform = comp1.transform
                mate.is_satisfied = True
    
    def _solve_distance_mate(self, mate: AssemblyMate):
        """Solve distance mate"""
        distance = mate.parameters.get('distance', 0.0)
        if len(mate.component_ids) >= 2 and distance > 0:
            comp1 = self.components.get(mate.component_ids[0])
            comp2 = self.components.get(mate.component_ids[1])
            
            if comp1 and comp2:
                # Simple distance constraint
                offset = Transform3D.translation(distance, 0, 0)
                comp2.transform = comp1.transform.compose(offset)
                mate.is_satisfied = True
    
    def generate_bom(self) -> Dict[str, Any]:
        """Generate Bill of Materials"""
        bom_items = []
        
        for i, (comp_id, component) in enumerate(self.components.items()):
            if component.state == ComponentState.RESOLVED:
                bom_items.append({
                    'item': i + 1,
                    'part_number': f'PART-{i+1:03d}',
                    'description': component.name,
                    'quantity': 1,
                    'source': component.source_path
                })
        
        return {
            'items': bom_items,
            'total_items': len(bom_items),
            'generated_date': 'now'
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get assembly statistics"""
        return {
            'total_components': len(self.components),
            'total_mates': len(self.mates),
            'resolved_components': sum(1 for c in self.components.values() 
                                     if c.state == ComponentState.RESOLVED),
            'satisfied_mates': sum(1 for m in self.mates.values() if m.is_satisfied)
        }
    
    def shutdown(self):
        """Shutdown assembly manager"""
        self.logger.info("Assembly Manager shutdown")