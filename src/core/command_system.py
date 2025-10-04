"""
Command System Implementation

Implements the Command Pattern for all user actions in OmniCAD.
Provides comprehensive undo/redo functionality and command history tracking.
"""

import json
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import threading
import copy

from ..utils.event_system import EventSystem
from ..utils.logger import OmniLogger


class CommandResult:
    """Result of command execution"""
    def __init__(self, success: bool = True, message: str = "", data: Any = None):
        self.success = success
        self.message = message
        self.data = data
        self.timestamp = datetime.now()
    
    def __bool__(self):
        return self.success


class Command(ABC):
    """
    Abstract base class for all commands in OmniCAD.
    
    All user actions (create feature, modify parameters, etc.) are implemented
    as commands that can be executed, undone, and redone.
    """
    
    def __init__(self, name: str, description: str = ""):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.timestamp = datetime.now()
        self.executed = False
        self.undone = False
        
        # Store state for undo/redo
        self.before_state: Optional[Any] = None
        self.after_state: Optional[Any] = None
        
        # Parameters passed to command
        self.parameters: Dict[str, Any] = {}
        
        # Result of execution
        self.result: Optional[CommandResult] = None
    
    @abstractmethod
    def execute(self, context: Any) -> CommandResult:
        """Execute the command"""
        pass
    
    @abstractmethod
    def undo(self, context: Any) -> CommandResult:
        """Undo the command"""
        pass
    
    def redo(self, context: Any) -> CommandResult:
        """Redo the command (default implementation re-executes)"""
        return self.execute(context)
    
    def can_undo(self) -> bool:
        """Check if command can be undone"""
        return self.executed and not self.undone
    
    def can_redo(self) -> bool:
        """Check if command can be redone"""
        return self.executed and self.undone
    
    def get_display_name(self) -> str:
        """Get display name for UI"""
        return self.name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'executed': self.executed,
            'undone': self.undone,
            'parameters': self.parameters
        }


# Specific command implementations

class CreateFeatureCommand(Command):
    """Command to create a new feature"""
    
    def __init__(self, parent_id: str, feature_type: str, name: str = "", parameters: Dict[str, Any] = None):
        super().__init__(f"Create {feature_type}", f"Create {feature_type} feature")
        self.parent_id = parent_id
        self.feature_type = feature_type
        self.feature_name = name
        self.feature_parameters = parameters or {}
        self.created_feature_id: Optional[str] = None
    
    def execute(self, context: Any) -> CommandResult:
        try:
            # Get the feature tree from context
            feature_tree = context.feature_tree
            
            from .feature_tree import FeatureType
            
            # Create the feature
            self.created_feature_id = feature_tree.add_feature(
                parent_id=self.parent_id,
                feature_type=FeatureType(self.feature_type),
                name=self.feature_name,
                parameters=self.feature_parameters
            )
            
            self.executed = True
            self.undone = False
            
            return CommandResult(True, f"Created feature: {self.feature_name}")
            
        except Exception as e:
            return CommandResult(False, f"Failed to create feature: {str(e)}")
    
    def undo(self, context: Any) -> CommandResult:
        try:
            if not self.created_feature_id:
                return CommandResult(False, "No feature to undo")
            
            # Remove the feature
            feature_tree = context.feature_tree
            feature_tree.remove_feature(self.created_feature_id)
            
            self.undone = True
            
            return CommandResult(True, f"Undone: Create {self.feature_name}")
            
        except Exception as e:
            return CommandResult(False, f"Failed to undo feature creation: {str(e)}")


class UpdateFeatureCommand(Command):
    """Command to update feature parameters"""
    
    def __init__(self, feature_id: str, parameter_name: str, old_value: Any, new_value: Any):
        super().__init__("Update Parameter", f"Update {parameter_name}")
        self.feature_id = feature_id
        self.parameter_name = parameter_name
        self.old_value = old_value
        self.new_value = new_value
    
    def execute(self, context: Any) -> CommandResult:
        try:
            feature_tree = context.feature_tree
            
            # Update the parameter
            feature_tree.update_feature(
                self.feature_id,
                **{f"parameters.{self.parameter_name}": self.new_value}
            )
            
            self.executed = True
            self.undone = False
            
            return CommandResult(True, f"Updated {self.parameter_name}")
            
        except Exception as e:
            return CommandResult(False, f"Failed to update parameter: {str(e)}")
    
    def undo(self, context: Any) -> CommandResult:
        try:
            feature_tree = context.feature_tree
            
            # Restore old value
            feature_tree.update_feature(
                self.feature_id,
                **{f"parameters.{self.parameter_name}": self.old_value}
            )
            
            self.undone = True
            
            return CommandResult(True, f"Undone: Update {self.parameter_name}")
            
        except Exception as e:
            return CommandResult(False, f"Failed to undo parameter update: {str(e)}")


class DeleteFeatureCommand(Command):
    """Command to delete a feature"""
    
    def __init__(self, feature_id: str):
        super().__init__("Delete Feature", "Delete feature")
        self.feature_id = feature_id
        self.saved_feature_data: Optional[Dict[str, Any]] = None
        self.saved_parent_id: Optional[str] = None
    
    def execute(self, context: Any) -> CommandResult:
        try:
            feature_tree = context.feature_tree
            
            # Save feature data for undo
            feature = feature_tree.get_node(self.feature_id)
            if not feature:
                return CommandResult(False, "Feature not found")
            
            self.saved_feature_data = feature.to_dict()
            self.saved_parent_id = feature.parent.id if feature.parent else None
            
            # Delete the feature
            feature_tree.remove_feature(self.feature_id)
            
            self.executed = True
            self.undone = False
            
            return CommandResult(True, f"Deleted feature: {feature.name}")
            
        except Exception as e:
            return CommandResult(False, f"Failed to delete feature: {str(e)}")
    
    def undo(self, context: Any) -> CommandResult:
        try:
            if not self.saved_feature_data or not self.saved_parent_id:
                return CommandResult(False, "No data to restore")
            
            feature_tree = context.feature_tree
            
            # Restore the feature
            from .feature_tree import FeatureNode
            restored_feature = FeatureNode.from_dict(self.saved_feature_data)
            
            parent = feature_tree.get_node(self.saved_parent_id)
            if parent:
                parent.add_child(restored_feature)
                feature_tree.nodes[restored_feature.id] = restored_feature
            
            self.undone = True
            
            return CommandResult(True, "Undone: Delete feature")
            
        except Exception as e:
            return CommandResult(False, f"Failed to undo feature deletion: {str(e)}")


class MacroCommand(Command):
    """Command that executes multiple sub-commands"""
    
    def __init__(self, name: str, commands: List[Command]):
        super().__init__(name, f"Execute {len(commands)} commands")
        self.commands = commands
        self.executed_commands: List[Command] = []
    
    def execute(self, context: Any) -> CommandResult:
        try:
            self.executed_commands.clear()
            
            for command in self.commands:
                result = command.execute(context)
                if not result.success:
                    # Rollback executed commands
                    for executed_cmd in reversed(self.executed_commands):
                        executed_cmd.undo(context)
                    return CommandResult(False, f"Failed at command: {command.name}")
                
                self.executed_commands.append(command)
            
            self.executed = True
            self.undone = False
            
            return CommandResult(True, f"Executed {len(self.commands)} commands")
            
        except Exception as e:
            return CommandResult(False, f"Macro command failed: {str(e)}")
    
    def undo(self, context: Any) -> CommandResult:
        try:
            # Undo in reverse order
            for command in reversed(self.executed_commands):
                result = command.undo(context)
                if not result.success:
                    return CommandResult(False, f"Failed to undo: {command.name}")
            
            self.undone = True
            
            return CommandResult(True, "Undone macro command")
            
        except Exception as e:
            return CommandResult(False, f"Failed to undo macro: {str(e)}")


class CommandManager:
    """
    Manages command execution, undo/redo history, and command registration.
    """
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.history: List[Command] = []
        self.current_index = -1
        
        # Command registry
        self.command_registry: Dict[str, type] = {}
        self.command_factories: Dict[str, Callable] = {}
        
        # Event system
        self.event_system = EventSystem()
        self.logger = OmniLogger("CommandManager")
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Register built-in commands
        self._register_builtin_commands()
    
    async def initialize(self):
        """Initialize the command manager"""
        self.logger.info("Initializing Command Manager")
    
    def _register_builtin_commands(self):
        """Register built-in command types"""
        self.register_command("create_feature", CreateFeatureCommand)
        self.register_command("update_feature", UpdateFeatureCommand)
        self.register_command("delete_feature", DeleteFeatureCommand)
        self.register_command("macro", MacroCommand)
    
    def register_command(self, name: str, command_class: type):
        """Register a command class"""
        self.command_registry[name] = command_class
        self.logger.debug(f"Registered command: {name}")
    
    def register_command_factory(self, name: str, factory: Callable):
        """Register a command factory function"""
        self.command_factories[name] = factory
        self.logger.debug(f"Registered command factory: {name}")
    
    def create_command(self, command_type: str, **kwargs) -> Optional[Command]:
        """Create a command instance"""
        try:
            # Try factory first
            if command_type in self.command_factories:
                return self.command_factories[command_type](**kwargs)
            
            # Try registered class
            if command_type in self.command_registry:
                command_class = self.command_registry[command_type]
                return command_class(**kwargs)
            
            self.logger.error(f"Unknown command type: {command_type}")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to create command {command_type}: {str(e)}")
            return None
    
    def execute(self, command: Union[Command, str], context: Any = None, **kwargs) -> CommandResult:
        """Execute a command"""
        with self._lock:
            # Create command if string is passed
            if isinstance(command, str):
                command = self.create_command(command, **kwargs)
                if not command:
                    return CommandResult(False, f"Failed to create command: {command}")
            
            try:
                # Get context (app instance)
                if context is None:
                    from .app import get_app
                    context = get_app()
                
                # Execute the command
                result = command.execute(context)
                
                if result.success:
                    # Add to history
                    self._add_to_history(command)
                    
                    # Emit event
                    self.event_system.emit('command_executed', {
                        'command_id': command.id,
                        'command_name': command.name,
                        'success': True
                    })
                    
                    self.logger.debug(f"Executed command: {command.name}")
                else:
                    self.logger.warning(f"Command failed: {command.name} - {result.message}")
                
                command.result = result
                return result
                
            except Exception as e:
                error_result = CommandResult(False, f"Command execution error: {str(e)}")
                command.result = error_result
                self.logger.error(f"Command execution failed: {command.name} - {str(e)}")
                return error_result
    
    def undo(self) -> CommandResult:
        """Undo the last command"""
        with self._lock:
            if self.current_index < 0:
                return CommandResult(False, "Nothing to undo")
            
            command = self.history[self.current_index]
            
            if not command.can_undo():
                return CommandResult(False, f"Cannot undo command: {command.name}")
            
            try:
                from .app import get_app
                context = get_app()
                
                result = command.undo(context)
                
                if result.success:
                    self.current_index -= 1
                    
                    # Emit event
                    self.event_system.emit('command_undone', {
                        'command_id': command.id,
                        'command_name': command.name
                    })
                    
                    self.logger.debug(f"Undone command: {command.name}")
                
                return result
                
            except Exception as e:
                error_result = CommandResult(False, f"Undo failed: {str(e)}")
                self.logger.error(f"Undo failed: {command.name} - {str(e)}")
                return error_result
    
    def redo(self) -> CommandResult:
        """Redo the next command"""
        with self._lock:
            if self.current_index >= len(self.history) - 1:
                return CommandResult(False, "Nothing to redo")
            
            command = self.history[self.current_index + 1]
            
            if not command.can_redo():
                return CommandResult(False, f"Cannot redo command: {command.name}")
            
            try:
                from .app import get_app
                context = get_app()
                
                result = command.redo(context)
                
                if result.success:
                    self.current_index += 1
                    
                    # Emit event
                    self.event_system.emit('command_redone', {
                        'command_id': command.id,
                        'command_name': command.name
                    })
                    
                    self.logger.debug(f"Redone command: {command.name}")
                
                return result
                
            except Exception as e:
                error_result = CommandResult(False, f"Redo failed: {str(e)}")
                self.logger.error(f"Redo failed: {command.name} - {str(e)}")
                return error_result
    
    def can_undo(self) -> bool:
        """Check if undo is possible"""
        return (self.current_index >= 0 and 
                self.current_index < len(self.history) and
                self.history[self.current_index].can_undo())
    
    def can_redo(self) -> bool:
        """Check if redo is possible"""
        return (self.current_index < len(self.history) - 1 and
                self.history[self.current_index + 1].can_redo())
    
    def get_undo_description(self) -> Optional[str]:
        """Get description of command that would be undone"""
        if self.can_undo():
            return f"Undo {self.history[self.current_index].get_display_name()}"
        return None
    
    def get_redo_description(self) -> Optional[str]:
        """Get description of command that would be redone"""
        if self.can_redo():
            return f"Redo {self.history[self.current_index + 1].get_display_name()}"
        return None
    
    def clear_history(self):
        """Clear the command history"""
        with self._lock:
            self.history.clear()
            self.current_index = -1
            self.logger.info("Command history cleared")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get command history for display"""
        return [cmd.to_dict() for cmd in self.history]
    
    def _add_to_history(self, command: Command):
        """Add a command to the history"""
        # Remove any commands after current index (redo chain broken)
        if self.current_index < len(self.history) - 1:
            self.history = self.history[:self.current_index + 1]
        
        # Add new command
        self.history.append(command)
        self.current_index += 1
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.current_index -= 1