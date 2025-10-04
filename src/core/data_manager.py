"""
Data Manager Implementation

Handles all data persistence operations including:
- Project file management (.omniproject format)
- Import/export of standard formats (STEP, IGES, STL, etc.)
- Local storage management
- Backup and recovery
"""

import json
import os
import zipfile
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Union, IO
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import base64

from ..utils.logger import OmniLogger
from ..utils.event_system import EventSystem


@dataclass
class FileMetadata:
    """Metadata for files in the project"""
    name: str
    size: int
    checksum: str
    mime_type: str
    created: str
    modified: str


class OmniProjectFormat:
    """
    OmniCAD native project format (.omniproject)
    
    This is a ZIP file containing:
    - project.json: Main project data
    - geometry/: Geometric data files
    - thumbnails/: Preview images
    - resources/: Additional resources (textures, etc.)
    """
    
    VERSION = "1.0"
    
    @staticmethod
    def create_project_structure(project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create the standard project structure"""
        return {
            'format_version': OmniProjectFormat.VERSION,
            'created': datetime.now().isoformat(),
            'application': 'OmniCAD',
            'application_version': '0.1.0',
            'project': project_data,
            'metadata': {
                'title': project_data.get('name', 'Untitled'),
                'description': project_data.get('metadata', {}).get('description', ''),
                'author': project_data.get('metadata', {}).get('author', ''),
                'keywords': [],
                'units': project_data.get('metadata', {}).get('units', 'mm')
            },
            'files': {},
            'dependencies': [],
            'checksums': {}
        }


class DataManager:
    """
    Manages all data operations for OmniCAD.
    
    Handles project persistence, file I/O, and data format conversions.
    """
    
    def __init__(self):
        self.logger = OmniLogger("DataManager")
        self.event_system = EventSystem()
        
        # File format handlers
        self.format_handlers = {}
        self.import_handlers = {}
        self.export_handlers = {}
        
        # Supported formats
        self.supported_import_formats = [
            '.omniproject',  # Native format
            '.step', '.stp',  # STEP files
            '.iges', '.igs',  # IGES files
            '.stl',          # STL files
            '.obj',          # OBJ files
            '.ply',          # PLY files
            '.dxf',          # DXF files
            '.dwg',          # DWG files (if supported)
            '.3mf',          # 3MF files
            '.amf'           # AMF files
        ]
        
        self.supported_export_formats = [
            '.omniproject',  # Native format
            '.step', '.stp',  # STEP files
            '.iges', '.igs',  # IGES files
            '.stl',          # STL files
            '.obj',          # OBJ files
            '.ply',          # PLY files
            '.dxf',          # DXF files
            '.gcode',        # G-code files
            '.nc',           # NC files
            '.json',         # JSON export
            '.pdf',          # PDF drawings
            '.png', '.jpg'   # Image exports
        ]
        
        # Current project info
        self.current_project_path: Optional[str] = None
        self.auto_save_enabled = True
        self.auto_save_interval = 300  # 5 minutes
        
        # Backup settings
        self.backup_enabled = True
        self.max_backups = 10
        
    async def initialize(self):
        """Initialize the data manager"""
        self.logger.info("Initializing Data Manager")
        
        # Register format handlers
        self._register_format_handlers()
        
        # Set up auto-save if enabled
        if self.auto_save_enabled:
            self._setup_auto_save()
    
    def _register_format_handlers(self):
        """Register handlers for different file formats"""
        # Native format
        self.import_handlers['.omniproject'] = self._import_omniproject
        self.export_handlers['.omniproject'] = self._export_omniproject
        
        # STEP format
        self.import_handlers['.step'] = self._import_step
        self.import_handlers['.stp'] = self._import_step
        self.export_handlers['.step'] = self._export_step
        self.export_handlers['.stp'] = self._export_step
        
        # STL format
        self.import_handlers['.stl'] = self._import_stl
        self.export_handlers['.stl'] = self._export_stl
        
        # JSON format
        self.export_handlers['.json'] = self._export_json
        
        # G-code format
        self.export_handlers['.gcode'] = self._export_gcode
        self.export_handlers['.nc'] = self._export_gcode
    
    async def save_document(self, document: Dict[str, Any], file_path: str = None) -> str:
        """Save a document to file"""
        try:
            # Determine file path
            if not file_path:
                if self.current_project_path:
                    file_path = self.current_project_path
                else:
                    # Generate default filename
                    doc_name = document.get('name', 'untitled')
                    file_path = f"{doc_name}.omniproject"
            
            # Ensure .omniproject extension for native format
            if not file_path.endswith('.omniproject'):
                file_path += '.omniproject'
            
            # Create backup if file exists
            if os.path.exists(file_path) and self.backup_enabled:
                await self._create_backup(file_path)
            
            # Save the document
            await self._save_omniproject(document, file_path)
            
            self.current_project_path = file_path
            
            self.logger.info(f"Document saved to: {file_path}")
            
            # Emit event
            self.event_system.emit('document_saved', {
                'file_path': file_path,
                'document_id': document.get('id'),
                'timestamp': datetime.now().isoformat()
            })
            
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to save document: {str(e)}")
            raise
    
    async def load_document(self, file_path: str) -> Dict[str, Any]:
        """Load a document from file"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Determine format from extension
            _, ext = os.path.splitext(file_path.lower())
            
            if ext not in self.import_handlers:
                raise ValueError(f"Unsupported file format: {ext}")
            
            # Load using appropriate handler
            document = await self.import_handlers[ext](file_path)
            
            self.current_project_path = file_path
            
            self.logger.info(f"Document loaded from: {file_path}")
            
            # Emit event
            self.event_system.emit('document_loaded', {
                'file_path': file_path,
                'document_id': document.get('id'),
                'timestamp': datetime.now().isoformat()
            })
            
            return document
            
        except Exception as e:
            self.logger.error(f"Failed to load document: {str(e)}")
            raise
    
    async def import_file(self, file_path: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Import a file and return imported data"""
        try:
            _, ext = os.path.splitext(file_path.lower())
            
            if ext not in self.import_handlers:
                raise ValueError(f"Import not supported for format: {ext}")
            
            # Import using appropriate handler
            imported_data = await self.import_handlers[ext](file_path, options or {})
            
            self.logger.info(f"Imported file: {file_path}")
            
            # Emit event
            self.event_system.emit('file_imported', {
                'file_path': file_path,
                'format': ext,
                'timestamp': datetime.now().isoformat()
            })
            
            return imported_data
            
        except Exception as e:
            self.logger.error(f"Failed to import file: {str(e)}")
            raise
    
    async def export_file(self, file_path: str, data: Dict[str, Any], options: Dict[str, Any] = None) -> str:
        """Export data to a file"""
        try:
            _, ext = os.path.splitext(file_path.lower())
            
            if ext not in self.export_handlers:
                raise ValueError(f"Export not supported for format: {ext}")
            
            # Export using appropriate handler
            exported_path = await self.export_handlers[ext](file_path, data, options or {})
            
            self.logger.info(f"Exported to: {exported_path}")
            
            # Emit event
            self.event_system.emit('file_exported', {
                'file_path': exported_path,
                'format': ext,
                'timestamp': datetime.now().isoformat()
            })
            
            return exported_path
            
        except Exception as e:
            self.logger.error(f"Failed to export file: {str(e)}")
            raise
    
    # Format-specific handlers
    
    async def _save_omniproject(self, document: Dict[str, Any], file_path: str):
        """Save in native OmniProject format"""
        # Create project structure
        project_data = OmniProjectFormat.create_project_structure(document)
        
        # Create temporary directory for assembly
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write main project file
            project_file = os.path.join(temp_dir, 'project.json')
            with open(project_file, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, indent=2, ensure_ascii=False)
            
            # Create directory structure
            os.makedirs(os.path.join(temp_dir, 'geometry'), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, 'thumbnails'), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, 'resources'), exist_ok=True)
            
            # Add geometry files if any
            # TODO: Implement geometry serialization
            
            # Create ZIP file
            with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path_in_zip = os.path.relpath(
                            os.path.join(root, file), temp_dir
                        )
                        zipf.write(os.path.join(root, file), file_path_in_zip)
    
    async def _import_omniproject(self, file_path: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Import OmniProject format"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP file
            with zipfile.ZipFile(file_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Read main project file
            project_file = os.path.join(temp_dir, 'project.json')
            if not os.path.exists(project_file):
                raise ValueError("Invalid OmniProject file: missing project.json")
            
            with open(project_file, 'r', encoding='utf-8') as f:
                project_data = json.load(f)
            
            # Validate format version
            format_version = project_data.get('format_version', '1.0')
            if format_version != OmniProjectFormat.VERSION:
                self.logger.warning(f"Format version mismatch: {format_version} vs {OmniProjectFormat.VERSION}")
            
            # Extract project document
            document = project_data.get('project', {})
            
            # TODO: Load geometry files
            
            return document
    
    async def _export_omniproject(self, file_path: str, data: Dict[str, Any], options: Dict[str, Any] = None) -> str:
        """Export to OmniProject format"""
        await self._save_omniproject(data, file_path)
        return file_path
    
    async def _import_step(self, file_path: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Import STEP file"""
        # TODO: Implement STEP import using OpenCASCADE
        self.logger.warning("STEP import not yet implemented")
        
        # Create placeholder document
        return {
            'id': 'imported_step',
            'name': os.path.basename(file_path),
            'imported_from': file_path,
            'format': 'STEP',
            'feature_tree': {},
            'metadata': {
                'description': f'Imported from {file_path}',
                'import_date': datetime.now().isoformat()
            }
        }
    
    async def _export_step(self, file_path: str, data: Dict[str, Any], options: Dict[str, Any] = None) -> str:
        """Export to STEP file"""
        # TODO: Implement STEP export using OpenCASCADE
        self.logger.warning("STEP export not yet implemented")
        
        # Create placeholder file
        with open(file_path, 'w') as f:
            f.write("ISO-10303-21;\n")
            f.write("HEADER;\n")
            f.write("FILE_DESCRIPTION(('OmniCAD Export'),'2;1');\n")
            f.write(f"FILE_NAME('{os.path.basename(file_path)}','{datetime.now().isoformat()}',('OmniCAD'),('OmniCAD'),'OmniCAD','OmniCAD','');\n")
            f.write("FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\n")
            f.write("ENDSEC;\n")
            f.write("DATA;\n")
            f.write("ENDSEC;\n")
            f.write("END-ISO-10303-21;\n")
        
        return file_path
    
    async def _import_stl(self, file_path: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Import STL file"""
        # TODO: Implement STL import
        self.logger.warning("STL import not yet implemented")
        
        return {
            'id': 'imported_stl',
            'name': os.path.basename(file_path),
            'imported_from': file_path,
            'format': 'STL',
            'feature_tree': {},
            'metadata': {
                'description': f'Imported STL mesh from {file_path}',
                'import_date': datetime.now().isoformat()
            }
        }
    
    async def _export_stl(self, file_path: str, data: Dict[str, Any], options: Dict[str, Any] = None) -> str:
        """Export to STL file"""
        # TODO: Implement STL export
        self.logger.warning("STL export not yet implemented")
        
        # Create placeholder binary STL
        with open(file_path, 'wb') as f:
            # STL header (80 bytes)
            header = b'OmniCAD STL Export' + b'\x00' * (80 - len('OmniCAD STL Export'))
            f.write(header)
            
            # Number of triangles (4 bytes)
            f.write((0).to_bytes(4, 'little'))
        
        return file_path
    
    async def _export_json(self, file_path: str, data: Dict[str, Any], options: Dict[str, Any] = None) -> str:
        """Export to JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        return file_path
    
    async def _export_gcode(self, file_path: str, data: Dict[str, Any], options: Dict[str, Any] = None) -> str:
        """Export G-code file"""
        # TODO: Implement G-code export from CAM data
        self.logger.warning("G-code export not yet implemented")
        
        # Create placeholder G-code
        with open(file_path, 'w') as f:
            f.write("; Generated by OmniCAD\n")
            f.write(f"; Date: {datetime.now().isoformat()}\n")
            f.write("; Material: \n")
            f.write("; Tool: \n")
            f.write("\n")
            f.write("G21 ; Set units to millimeters\n")
            f.write("G90 ; Use absolute positioning\n")
            f.write("G17 ; Select XY plane\n")
            f.write("\n")
            f.write("M30 ; Program end\n")
        
        return file_path
    
    async def _create_backup(self, file_path: str):
        """Create a backup of an existing file"""
        try:
            backup_dir = os.path.join(os.path.dirname(file_path), '.omnicad_backups')
            os.makedirs(backup_dir, exist_ok=True)
            
            # Generate backup filename with timestamp
            base_name = os.path.basename(file_path)
            name, ext = os.path.splitext(base_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{name}_{timestamp}{ext}"
            backup_path = os.path.join(backup_dir, backup_name)
            
            # Copy file
            shutil.copy2(file_path, backup_path)
            
            # Clean up old backups
            await self._cleanup_old_backups(backup_dir, base_name)
            
            self.logger.debug(f"Created backup: {backup_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create backup: {str(e)}")
    
    async def _cleanup_old_backups(self, backup_dir: str, base_name: str):
        """Clean up old backup files"""
        try:
            name, ext = os.path.splitext(base_name)
            pattern = f"{name}_"
            
            # Get all backup files for this base name
            backup_files = []
            for file in os.listdir(backup_dir):
                if file.startswith(pattern) and file.endswith(ext):
                    backup_files.append(os.path.join(backup_dir, file))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Remove excess backups
            for file_path in backup_files[self.max_backups:]:
                os.remove(file_path)
                self.logger.debug(f"Removed old backup: {file_path}")
                
        except Exception as e:
            self.logger.warning(f"Failed to cleanup old backups: {str(e)}")
    
    def _setup_auto_save(self):
        """Set up automatic saving"""
        # TODO: Implement auto-save timer
        self.logger.info(f"Auto-save enabled with {self.auto_save_interval}s interval")
    
    def get_supported_formats(self, operation: str = 'import') -> List[str]:
        """Get list of supported file formats"""
        if operation == 'import':
            return self.supported_import_formats.copy()
        elif operation == 'export':
            return self.supported_export_formats.copy()
        else:
            return list(set(self.supported_import_formats + self.supported_export_formats))
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()