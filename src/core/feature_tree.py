"""
Unified Feature Tree Implementation

The feature tree is the single source of truth for all application state.
It stores everything from geometric features to CAM operations to analysis results
in a hierarchical, queryable structure.
"""

import json
import uuid
from typing import Dict, Any, List, Optional, Union, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading

from ..utils.event_system import EventSystem
from ..utils.logger import OmniLogger


class FeatureType(Enum):
    """Enumeration of all possible feature types in OmniCAD"""
    # Document structure
    DOCUMENT = "document"
    PART = "part"
    ASSEMBLY = "assembly"
    
    # Geometric features
    SKETCH = "sketch"
    EXTRUDE = "extrude"
    REVOLVE = "revolve"
    SWEEP = "sweep"
    LOFT = "loft"
    FILLET = "fillet"
    CHAMFER = "chamfer"
    SHELL = "shell"
    DRAFT = "draft"
    
    # Pattern features
    LINEAR_PATTERN = "linear_pattern"
    CIRCULAR_PATTERN = "circular_pattern"
    MIRROR = "mirror"
    
    # Assembly features
    COMPONENT = "component"
    MATE = "mate"
    CONSTRAINT = "constraint"
    
    # CAM features
    CAM_SETUP = "cam_setup"
    STOCK = "stock"
    TOOLPATH = "toolpath"
    OPERATION = "operation"
    TOOL = "tool"
    
    # Analysis features
    STUDY = "study"
    LOAD = "load"
    FIXTURE = "fixture"
    MATERIAL = "material"
    MESH = "mesh"
    RESULT = "result"
    
    # Rendering features
    SCENE = "scene"
    CAMERA = "camera"
    LIGHT = "light"
    APPEARANCE = "appearance"
    
    # PLM features
    REVISION = "revision"
    WORKFLOW = "workflow"
    APPROVAL = "approval"
    
    # Utility features
    PLANE = "plane"
    AXIS = "axis"
    POINT = "point"
    COORDINATE_SYSTEM = "coordinate_system"


@dataclass
class FeatureData:
    """Container for feature-specific data"""
    type: FeatureType
    parameters: Dict[str, Any] = field(default_factory=dict)
    geometry: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'type': self.type.value,
            'parameters': self.parameters,
            'metadata': self.metadata,
            'properties': self.properties
            # Note: geometry is not serialized directly
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureData':
        """Create from dictionary"""
        return cls(
            type=FeatureType(data['type']),
            parameters=data.get('parameters', {}),
            metadata=data.get('metadata', {}),
            properties=data.get('properties', {})
        )


class FeatureNode:
    """
    A node in the unified feature tree.
    
    Each node represents a feature, operation, or data element in the system.
    The tree structure captures dependencies and hierarchical relationships.
    """
    
    def __init__(self, 
                 id: str = None,
                 name: str = "",
                 data: FeatureData = None,
                 parent: 'FeatureNode' = None):
        self.id = id or str(uuid.uuid4())
        self.name = name
        self.data = data or FeatureData(FeatureType.DOCUMENT)
        self.parent = parent
        self.children: List['FeatureNode'] = []
        
        # State tracking
        self.is_visible = True
        self.is_enabled = True
        self.is_suppressed = False
        self.is_selected = False
        
        # Timestamps
        self.created = datetime.now()
        self.modified = datetime.now()
        
        # Dependencies
        self.depends_on: List[str] = []  # Feature IDs this feature depends on
        self.dependents: List[str] = []  # Feature IDs that depend on this feature
        
        # Cached results
        self._cache: Dict[str, Any] = {}
        self._cache_valid = False
    
    def add_child(self, child: 'FeatureNode'):
        """Add a child node"""
        if child.parent:
            child.parent.remove_child(child)
        
        child.parent = self
        self.children.append(child)
        self._invalidate_cache()
    
    def remove_child(self, child: 'FeatureNode'):
        """Remove a child node"""
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            self._invalidate_cache()
    
    def find_child(self, id: str) -> Optional['FeatureNode']:
        """Find a child by ID"""
        for child in self.children:
            if child.id == id:
                return child
        return None
    
    def find_descendant(self, id: str) -> Optional['FeatureNode']:
        """Find a descendant by ID (recursive search)"""
        if self.id == id:
            return self
        
        for child in self.children:
            result = child.find_descendant(id)
            if result:
                return result
        
        return None
    
    def get_path(self) -> List[str]:
        """Get the path from root to this node"""
        path = []
        current = self
        while current:
            path.insert(0, current.id)
            current = current.parent
        return path
    
    def get_depth(self) -> int:
        """Get the depth of this node in the tree"""
        depth = 0
        current = self.parent
        while current:
            depth += 1
            current = current.parent
        return depth
    
    def is_ancestor_of(self, other: 'FeatureNode') -> bool:
        """Check if this node is an ancestor of another node"""
        current = other.parent
        while current:
            if current == self:
                return True
            current = current.parent
        return False
    
    def get_siblings(self) -> List['FeatureNode']:
        """Get sibling nodes"""
        if self.parent:
            return [child for child in self.parent.children if child != self]
        return []
    
    def update_data(self, **kwargs):
        """Update feature data"""
        for key, value in kwargs.items():
            if hasattr(self.data, key):
                setattr(self.data, key, value)
        
        self.modified = datetime.now()
        self._invalidate_cache()
    
    def add_dependency(self, feature_id: str):
        """Add a dependency on another feature"""
        if feature_id not in self.depends_on:
            self.depends_on.append(feature_id)
            self._invalidate_cache()
    
    def remove_dependency(self, feature_id: str):
        """Remove a dependency"""
        if feature_id in self.depends_on:
            self.depends_on.remove(feature_id)
            self._invalidate_cache()
    
    def _invalidate_cache(self):
        """Invalidate cached results"""
        self._cache_valid = False
        self._cache.clear()
        
        # Invalidate dependents as well
        # This would be handled by the feature tree
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'data': self.data.to_dict(),
            'is_visible': self.is_visible,
            'is_enabled': self.is_enabled,
            'is_suppressed': self.is_suppressed,
            'created': self.created.isoformat(),
            'modified': self.modified.isoformat(),
            'depends_on': self.depends_on,
            'children': [child.to_dict() for child in self.children]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureNode':
        """Create from dictionary"""
        node = cls(
            id=data['id'],
            name=data['name'],
            data=FeatureData.from_dict(data['data'])
        )
        
        node.is_visible = data.get('is_visible', True)
        node.is_enabled = data.get('is_enabled', True)
        node.is_suppressed = data.get('is_suppressed', False)
        node.created = datetime.fromisoformat(data['created'])
        node.modified = datetime.fromisoformat(data['modified'])
        node.depends_on = data.get('depends_on', [])
        
        # Recursively create children
        for child_data in data.get('children', []):
            child = cls.from_dict(child_data)
            node.add_child(child)
        
        return node


class FeatureTree:
    """
    The unified feature tree that serves as the single source of truth
    for all application state in OmniCAD.
    """
    
    def __init__(self):
        self.root: FeatureNode = None
        self.nodes: Dict[str, FeatureNode] = {}  # Fast lookup by ID
        self.logger = OmniLogger("FeatureTree")
        self.event_system = EventSystem()
        
        # Dependency tracking
        self.dependency_graph: Dict[str, List[str]] = {}
        
        # Change tracking
        self.modified_nodes: set = set()
        
        # Thread safety
        self._lock = threading.Lock()
    
    async def initialize(self):
        """Initialize the feature tree"""
        self.logger.info("Initializing Feature Tree")
        # Create root node if it doesn't exist
        if not self.root:
            self.create_document_tree()
    
    def create_document_tree(self) -> str:
        """Create a new document tree structure"""
        # Create root document node
        self.root = FeatureNode(
            name="Document",
            data=FeatureData(FeatureType.DOCUMENT)
        )
        
        # Add standard top-level nodes
        origin = FeatureNode(
            name="Origin",
            data=FeatureData(FeatureType.COORDINATE_SYSTEM, {
                'position': [0, 0, 0],
                'rotation': [0, 0, 0]
            })
        )
        
        features = FeatureNode(
            name="Features",
            data=FeatureData(FeatureType.DOCUMENT)
        )
        
        sketches = FeatureNode(
            name="Sketches",
            data=FeatureData(FeatureType.DOCUMENT)
        )
        
        materials = FeatureNode(
            name="Materials",
            data=FeatureData(FeatureType.DOCUMENT)
        )
        
        appearances = FeatureNode(
            name="Appearances",
            data=FeatureData(FeatureType.DOCUMENT)
        )
        
        # Add to tree
        self.root.add_child(origin)
        self.root.add_child(features)
        self.root.add_child(sketches)
        self.root.add_child(materials)
        self.root.add_child(appearances)
        
        # Update lookup table
        self._rebuild_lookup_table()
        
        self.logger.info(f"Created document tree with root: {self.root.id}")
        return self.root.id
    
    def add_feature(self, 
                   parent_id: str,
                   feature_type: FeatureType,
                   name: str = "",
                   parameters: Dict[str, Any] = None) -> str:
        """Add a new feature to the tree"""
        with self._lock:
            parent = self.get_node(parent_id)
            if not parent:
                raise ValueError(f"Parent node not found: {parent_id}")
            
            # Create feature data
            feature_data = FeatureData(
                type=feature_type,
                parameters=parameters or {},
                metadata={
                    'created_by': 'user',
                    'creation_method': 'interactive'
                }
            )
            
            # Create node
            feature_name = name or f"{feature_type.value}_{len(parent.children) + 1}"
            node = FeatureNode(
                name=feature_name,
                data=feature_data,
                parent=parent
            )
            
            # Add to parent
            parent.add_child(node)
            
            # Update lookup table
            self.nodes[node.id] = node
            
            # Emit event
            self.event_system.emit('feature_added', {
                'feature_id': node.id,
                'feature_type': feature_type.value,
                'parent_id': parent_id
            })
            
            self.logger.debug(f"Added feature: {feature_name} ({node.id})")
            return node.id
    
    def remove_feature(self, feature_id: str):
        """Remove a feature from the tree"""
        with self._lock:
            node = self.get_node(feature_id)
            if not node:
                raise ValueError(f"Feature not found: {feature_id}")
            
            if node == self.root:
                raise ValueError("Cannot remove root node")
            
            # Remove dependencies
            self._remove_dependencies(feature_id)
            
            # Remove from parent
            if node.parent:
                node.parent.remove_child(node)
            
            # Remove from lookup table
            self._remove_from_lookup(node)
            
            # Emit event
            self.event_system.emit('feature_removed', {
                'feature_id': feature_id
            })
            
            self.logger.debug(f"Removed feature: {feature_id}")
    
    def get_node(self, feature_id: str) -> Optional[FeatureNode]:
        """Get a node by ID"""
        return self.nodes.get(feature_id)
    
    def update_feature(self, feature_id: str, **kwargs):
        """Update a feature's data"""
        with self._lock:
            node = self.get_node(feature_id)
            if not node:
                raise ValueError(f"Feature not found: {feature_id}")
            
            node.update_data(**kwargs)
            self.modified_nodes.add(feature_id)
            
            # Emit event
            self.event_system.emit('feature_updated', {
                'feature_id': feature_id,
                'changes': kwargs
            })
    
    def get_features_by_type(self, feature_type: FeatureType) -> List[FeatureNode]:
        """Get all features of a specific type"""
        return [node for node in self.nodes.values() 
                if node.data.type == feature_type]
    
    def find_features(self, **criteria) -> List[FeatureNode]:
        """Find features matching criteria"""
        results = []
        
        for node in self.nodes.values():
            match = True
            
            # Check name
            if 'name' in criteria:
                if criteria['name'].lower() not in node.name.lower():
                    match = False
            
            # Check type
            if 'type' in criteria:
                if node.data.type != criteria['type']:
                    match = False
            
            # Check parameters
            if 'parameters' in criteria:
                for key, value in criteria['parameters'].items():
                    if key not in node.data.parameters:
                        match = False
                        break
                    if node.data.parameters[key] != value:
                        match = False
                        break
            
            if match:
                results.append(node)
        
        return results
    
    def get_dependency_order(self) -> List[str]:
        """Get features in dependency order (topological sort)"""
        # Simple topological sort implementation
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(node_id):
            if node_id in temp_visited:
                raise ValueError("Circular dependency detected")
            if node_id in visited:
                return
            
            temp_visited.add(node_id)
            node = self.get_node(node_id)
            if node:
                for dep_id in node.depends_on:
                    visit(dep_id)
            
            temp_visited.remove(node_id)
            visited.add(node_id)
            result.append(node_id)
        
        for node_id in self.nodes.keys():
            if node_id not in visited:
                visit(node_id)
        
        return result
    
    def validate_dependencies(self) -> List[str]:
        """Validate all dependencies and return any errors"""
        errors = []
        
        for node_id, node in self.nodes.items():
            for dep_id in node.depends_on:
                if dep_id not in self.nodes:
                    errors.append(f"Node {node_id} depends on missing node {dep_id}")
        
        # Check for circular dependencies
        try:
            self.get_dependency_order()
        except ValueError as e:
            errors.append(str(e))
        
        return errors
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize the entire tree to dictionary"""
        if not self.root:
            return {}
        
        return {
            'version': 1,
            'root': self.root.to_dict(),
            'metadata': {
                'serialized': datetime.now().isoformat(),
                'node_count': len(self.nodes)
            }
        }
    
    async def load_from_data(self, data: Dict[str, Any]):
        """Load tree from serialized data"""
        if not data or 'root' not in data:
            return
        
        # Clear existing tree
        self.root = None
        self.nodes.clear()
        
        # Reconstruct tree
        self.root = FeatureNode.from_dict(data['root'])
        self._rebuild_lookup_table()
        
        self.logger.info(f"Loaded feature tree with {len(self.nodes)} nodes")
    
    def _rebuild_lookup_table(self):
        """Rebuild the node lookup table"""
        self.nodes.clear()
        
        def add_to_lookup(node: FeatureNode):
            self.nodes[node.id] = node
            for child in node.children:
                add_to_lookup(child)
        
        if self.root:
            add_to_lookup(self.root)
    
    def _remove_from_lookup(self, node: FeatureNode):
        """Remove node and its children from lookup table"""
        if node.id in self.nodes:
            del self.nodes[node.id]
        
        for child in node.children:
            self._remove_from_lookup(child)
    
    def _remove_dependencies(self, feature_id: str):
        """Remove all dependencies related to a feature"""
        # Remove this feature from all dependency lists
        for node in self.nodes.values():
            if feature_id in node.depends_on:
                node.depends_on.remove(feature_id)
            if feature_id in node.dependents:
                node.dependents.remove(feature_id)