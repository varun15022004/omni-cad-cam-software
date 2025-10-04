"""
OmniCAD Core Application

This module provides the main application controller and orchestrates all subsystems.
It implements the unified feature tree, command system, and data management.
"""

import json
import uuid
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import threading
import asyncio

from ..utils.event_system import EventSystem
from ..utils.logger import OmniLogger
from .feature_tree import FeatureTree, FeatureNode
from .command_system import CommandManager, Command
from .data_manager import DataManager


@dataclass
class ApplicationState:
    """Complete application state container"""
    current_mode: str = "model"  # model, assembly, cam, simulation, rendering, plm, generative
    active_document: Optional[str] = None
    selected_features: List[str] = field(default_factory=list)
    viewport_config: Dict[str, Any] = field(default_factory=dict)
    ui_layout: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class OmniCADApp:
    """
    Main application controller for Project OmniCAD.
    
    This class orchestrates all modules and maintains the unified application state.
    It provides the central hub for communication between UI, geometry kernel,
    CAM system, analysis engines, and all other subsystems.
    """
    
    def __init__(self):
        # Core systems
        self.logger = OmniLogger("OmniCAD")
        self.event_system = EventSystem()
        self.feature_tree = FeatureTree()
        self.command_manager = CommandManager()
        self.data_manager = DataManager()
        
        # Application state
        self.state = ApplicationState()
        self.documents: Dict[str, Any] = {}
        self.modules: Dict[str, Any] = {}
        
        # Threading for heavy operations
        self.worker_thread = None
        self.is_running = False
        
        # Module loading flags
        self.loaded_modules = set()
        
        self.logger.info("OmniCAD Application initialized")
    
    async def initialize(self):
        """Initialize the application and load core modules"""
        try:
            self.logger.info("Starting OmniCAD initialization...")
            
            # Initialize core systems
            await self.feature_tree.initialize()
            await self.command_manager.initialize()
            await self.data_manager.initialize()
            
            # Set up event handlers
            self._setup_event_handlers()
            
            # Load essential modules
            await self._load_core_modules()
            
            # Initialize default document
            await self.create_new_document()
            
            self.is_running = True
            self.logger.info("OmniCAD initialization completed successfully")
            
            # Emit initialization complete event
            self.event_system.emit('app_initialized', {
                'timestamp': datetime.now().isoformat(),
                'version': '0.1.0',
                'modules_loaded': list(self.loaded_modules)
            })
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OmniCAD: {str(e)}")
            raise
    
    def _setup_event_handlers(self):
        """Set up core event handlers"""
        # Selection change events
        self.event_system.on('selection_changed', self._handle_selection_change)
        
        # Mode change events
        self.event_system.on('mode_changed', self._handle_mode_change)
        
        # Document events
        self.event_system.on('document_modified', self._handle_document_modified)
        
        # Command events
        self.event_system.on('command_executed', self._handle_command_executed)
        self.event_system.on('command_undone', self._handle_command_undone)
        
        # Error handling
        self.event_system.on('error', self._handle_error)
    
    async def _load_core_modules(self):
        """Load essential modules for basic functionality"""
        core_modules = [
            'geometry',    # Geometric modeling kernel
            'ui',         # User interface components
            'rendering'   # Basic rendering capabilities
        ]
        
        for module_name in core_modules:
            await self.load_module(module_name)
    
    async def load_module(self, module_name: str):
        """Dynamically load a module"""
        if module_name in self.loaded_modules:
            self.logger.debug(f"Module {module_name} already loaded")
            return
        
        try:
            self.logger.info(f"Loading module: {module_name}")
            
            # Import the module dynamically
            if module_name == 'geometry':
                from ..geometry import GeometryKernel
                self.modules[module_name] = GeometryKernel()
            elif module_name == 'ui':
                from ..ui import UIManager
                self.modules[module_name] = UIManager()
            elif module_name == 'rendering':
                from ..rendering import RenderingEngine
                self.modules[module_name] = RenderingEngine()
            elif module_name == 'cam':
                from ..cam import CAMEngine
                self.modules[module_name] = CAMEngine()
            elif module_name == 'analysis':
                from ..analysis import AnalysisEngine
                self.modules[module_name] = AnalysisEngine()
            elif module_name == 'assembly':
                from ..assembly import AssemblyManager
                self.modules[module_name] = AssemblyManager()
            elif module_name == 'plm':
                from ..plm import PLMManager
                self.modules[module_name] = PLMManager()
            elif module_name == 'specialized':
                from ..specialized import SpecializedTechManager
                self.modules[module_name] = SpecializedTechManager()
            else:
                raise ValueError(f"Unknown module: {module_name}")
            
            # Initialize the module
            if hasattr(self.modules[module_name], 'initialize'):
                await self.modules[module_name].initialize(self)
            
            self.loaded_modules.add(module_name)
            self.logger.info(f"Module {module_name} loaded successfully")
            
            # Emit module loaded event
            self.event_system.emit('module_loaded', {
                'module': module_name,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Failed to load module {module_name}: {str(e)}")
            raise
    
    async def create_new_document(self, name: str = None) -> str:
        """Create a new document"""
        doc_id = str(uuid.uuid4())
        doc_name = name or f"Untitled_{len(self.documents) + 1}"
        
        # Create document structure
        document = {
            'id': doc_id,
            'name': doc_name,
            'created': datetime.now().isoformat(),
            'modified': datetime.now().isoformat(),
            'feature_tree': self.feature_tree.create_document_tree(),
            'metadata': {
                'author': 'OmniCAD User',
                'description': '',
                'units': 'mm',
                'precision': 0.01
            },
            'version': 1
        }
        
        self.documents[doc_id] = document
        self.state.active_document = doc_id
        
        self.logger.info(f"Created new document: {doc_name} ({doc_id})")
        
        # Emit event
        self.event_system.emit('document_created', {
            'document_id': doc_id,
            'document_name': doc_name
        })
        
        return doc_id
    
    async def open_document(self, file_path: str) -> str:
        """Open a document from file"""
        try:
            doc_data = await self.data_manager.load_document(file_path)
            doc_id = doc_data['id']
            
            self.documents[doc_id] = doc_data
            self.state.active_document = doc_id
            
            # Restore feature tree
            await self.feature_tree.load_from_data(doc_data['feature_tree'])
            
            self.logger.info(f"Opened document: {doc_data['name']} from {file_path}")
            
            # Emit event
            self.event_system.emit('document_opened', {
                'document_id': doc_id,
                'file_path': file_path
            })
            
            return doc_id
            
        except Exception as e:
            self.logger.error(f"Failed to open document from {file_path}: {str(e)}")
            raise
    
    async def save_document(self, doc_id: str = None, file_path: str = None) -> str:
        """Save a document to file"""
        doc_id = doc_id or self.state.active_document
        if not doc_id or doc_id not in self.documents:
            raise ValueError("No document to save")
        
        document = self.documents[doc_id]
        document['modified'] = datetime.now().isoformat()
        document['version'] += 1
        
        # Update feature tree data
        document['feature_tree'] = self.feature_tree.serialize()
        
        saved_path = await self.data_manager.save_document(document, file_path)
        
        self.logger.info(f"Saved document: {document['name']} to {saved_path}")
        
        # Emit event
        self.event_system.emit('document_saved', {
            'document_id': doc_id,
            'file_path': saved_path
        })
        
        return saved_path
    
    def execute_command(self, command_name: str, **kwargs) -> Any:
        """Execute a command through the command manager"""
        try:
            result = self.command_manager.execute(command_name, **kwargs)
            
            # Mark document as modified
            if self.state.active_document:
                self.documents[self.state.active_document]['modified'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Command execution failed: {command_name} - {str(e)}")
            raise
    
    def undo(self):
        """Undo the last command"""
        return self.command_manager.undo()
    
    def redo(self):
        """Redo the last undone command"""
        return self.command_manager.redo()
    
    def set_mode(self, mode: str):
        """Change application mode"""
        if mode != self.state.current_mode:
            old_mode = self.state.current_mode
            self.state.current_mode = mode
            
            self.logger.info(f"Mode changed: {old_mode} -> {mode}")
            
            # Emit mode change event
            self.event_system.emit('mode_changed', {
                'old_mode': old_mode,
                'new_mode': mode
            })
    
    def select_features(self, feature_ids: List[str]):
        """Select features in the feature tree"""
        self.state.selected_features = feature_ids
        
        # Emit selection change event
        self.event_system.emit('selection_changed', {
            'selected_features': feature_ids
        })
    
    def get_active_document(self) -> Optional[Dict[str, Any]]:
        """Get the currently active document"""
        if self.state.active_document:
            return self.documents.get(self.state.active_document)
        return None
    
    def get_module(self, module_name: str) -> Optional[Any]:
        """Get a loaded module"""
        return self.modules.get(module_name)
    
    # Event handlers
    def _handle_selection_change(self, event_data):
        """Handle selection change events"""
        selected_features = event_data['selected_features']
        self.logger.debug(f"Selection changed to: {selected_features}")
    
    def _handle_mode_change(self, event_data):
        """Handle mode change events"""
        new_mode = event_data['new_mode']
        
        # Load required modules for the new mode
        if new_mode == 'cam' and 'cam' not in self.loaded_modules:
            asyncio.create_task(self.load_module('cam'))
        elif new_mode == 'simulation' and 'analysis' not in self.loaded_modules:
            asyncio.create_task(self.load_module('analysis'))
        elif new_mode == 'assembly' and 'assembly' not in self.loaded_modules:
            asyncio.create_task(self.load_module('assembly'))
        elif new_mode == 'plm' and 'plm' not in self.loaded_modules:
            asyncio.create_task(self.load_module('plm'))
    
    def _handle_document_modified(self, event_data):
        """Handle document modification events"""
        self.logger.debug("Document modified")
    
    def _handle_command_executed(self, event_data):
        """Handle command execution events"""
        command_name = event_data.get('command_name', 'Unknown')
        self.logger.debug(f"Command executed: {command_name}")
    
    def _handle_command_undone(self, event_data):
        """Handle command undo events"""
        command_name = event_data.get('command_name', 'Unknown')
        self.logger.debug(f"Command undone: {command_name}")
    
    def _handle_error(self, event_data):
        """Handle error events"""
        error_message = event_data.get('message', 'Unknown error')
        self.logger.error(f"Application error: {error_message}")
    
    def shutdown(self):
        """Shutdown the application"""
        self.logger.info("Shutting down OmniCAD...")
        
        self.is_running = False
        
        # Cleanup modules
        for module_name, module in self.modules.items():
            if hasattr(module, 'shutdown'):
                try:
                    module.shutdown()
                except Exception as e:
                    self.logger.error(f"Error shutting down module {module_name}: {str(e)}")
        
        # Emit shutdown event
        self.event_system.emit('app_shutdown', {
            'timestamp': datetime.now().isoformat()
        })
        
        self.logger.info("OmniCAD shutdown completed")


# Global application instance
app_instance = None

def get_app() -> OmniCADApp:
    """Get the global application instance"""
    global app_instance
    if app_instance is None:
        app_instance = OmniCADApp()
    return app_instance