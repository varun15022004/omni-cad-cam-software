"""
Logging System for OmniCAD

Provides comprehensive logging capabilities with multiple output targets,
log levels, and performance monitoring.
"""

import logging
import json
import os
import sys
from typing import Dict, Any, List, Optional, TextIO
from datetime import datetime
from enum import Enum
import threading
import queue
import traceback


class LogLevel(Enum):
    """Log levels for OmniCAD"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogEntry:
    """Represents a single log entry"""
    
    def __init__(self, level: LogLevel, message: str, logger_name: str, 
                 module: str = None, function: str = None, line: int = None,
                 extra_data: Dict[str, Any] = None):
        self.timestamp = datetime.now()
        self.level = level
        self.message = message
        self.logger_name = logger_name
        self.module = module
        self.function = function
        self.line = line
        self.extra_data = extra_data or {}
        self.thread_id = threading.get_ident()
        self.thread_name = threading.current_thread().name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'message': self.message,
            'logger': self.logger_name,
            'module': self.module,
            'function': self.function,
            'line': self.line,
            'thread_id': self.thread_id,
            'thread_name': self.thread_name,
            'extra': self.extra_data
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)


class LogHandler:
    """Base class for log handlers"""
    
    def __init__(self, level: LogLevel = LogLevel.INFO):
        self.level = level
        self.enabled = True
    
    def can_handle(self, entry: LogEntry) -> bool:
        """Check if this handler should process the log entry"""
        if not self.enabled:
            return False
        
        level_order = {
            LogLevel.DEBUG: 1,
            LogLevel.INFO: 2,
            LogLevel.WARNING: 3,
            LogLevel.ERROR: 4,
            LogLevel.CRITICAL: 5
        }
        
        return level_order[entry.level] >= level_order[self.level]
    
    def handle(self, entry: LogEntry):
        """Handle a log entry"""
        pass


class ConsoleHandler(LogHandler):
    """Outputs logs to console"""
    
    def __init__(self, level: LogLevel = LogLevel.INFO, 
                 stream: TextIO = None, colored: bool = True):
        super().__init__(level)
        self.stream = stream or sys.stdout
        self.colored = colored
        
        # ANSI color codes
        self.colors = {
            LogLevel.DEBUG: '\033[36m',    # Cyan
            LogLevel.INFO: '\033[32m',     # Green
            LogLevel.WARNING: '\033[33m',  # Yellow
            LogLevel.ERROR: '\033[31m',    # Red
            LogLevel.CRITICAL: '\033[35m'  # Magenta
        }
        self.reset_color = '\033[0m'
    
    def handle(self, entry: LogEntry):
        """Output log entry to console"""
        if not self.can_handle(entry):
            return
        
        # Format timestamp
        timestamp = entry.timestamp.strftime("%H:%M:%S.%f")[:-3]
        
        # Format message
        if self.colored and entry.level in self.colors:
            color = self.colors[entry.level]
            level_str = f"{color}[{entry.level.value:8}]{self.reset_color}"
        else:
            level_str = f"[{entry.level.value:8}]"
        
        # Include module/function info for debug
        location = ""
        if entry.level == LogLevel.DEBUG and (entry.module or entry.function):
            location = f" ({entry.module or ''}:{entry.function or ''})"
        
        message = f"{timestamp} {level_str} [{entry.logger_name:12}]{location} {entry.message}"
        
        print(message, file=self.stream)
        
        # Flush immediately for errors
        if entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self.stream.flush()


class FileHandler(LogHandler):
    """Outputs logs to file"""
    
    def __init__(self, file_path: str, level: LogLevel = LogLevel.DEBUG,
                 max_size: int = 10 * 1024 * 1024, backup_count: int = 5):
        super().__init__(level)
        self.file_path = file_path
        self.max_size = max_size
        self.backup_count = backup_count
        self.current_size = 0
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Open file
        self.file = open(file_path, 'a', encoding='utf-8')
        self.current_size = self.file.tell()
    
    def handle(self, entry: LogEntry):
        """Write log entry to file"""
        if not self.can_handle(entry):
            return
        
        # Format entry as JSON
        log_line = entry.to_json() + '\n'
        
        # Check if rotation is needed
        if self.current_size + len(log_line.encode('utf-8')) > self.max_size:
            self._rotate_file()
        
        # Write to file
        self.file.write(log_line)
        self.file.flush()
        self.current_size += len(log_line.encode('utf-8'))
    
    def _rotate_file(self):
        """Rotate log file when it gets too large"""
        self.file.close()
        
        # Rotate existing files
        for i in range(self.backup_count - 1, 0, -1):
            old_file = f"{self.file_path}.{i}"
            new_file = f"{self.file_path}.{i + 1}"
            
            if os.path.exists(old_file):
                if os.path.exists(new_file):
                    os.remove(new_file)
                os.rename(old_file, new_file)
        
        # Move current file to .1
        if os.path.exists(self.file_path):
            backup_file = f"{self.file_path}.1"
            if os.path.exists(backup_file):
                os.remove(backup_file)
            os.rename(self.file_path, backup_file)
        
        # Open new file
        self.file = open(self.file_path, 'w', encoding='utf-8')
        self.current_size = 0
    
    def close(self):
        """Close file handler"""
        if self.file:
            self.file.close()


class MemoryHandler(LogHandler):
    """Stores logs in memory for runtime inspection"""
    
    def __init__(self, level: LogLevel = LogLevel.DEBUG, max_entries: int = 1000):
        super().__init__(level)
        self.max_entries = max_entries
        self.entries: List[LogEntry] = []
        self._lock = threading.Lock()
    
    def handle(self, entry: LogEntry):
        """Store log entry in memory"""
        if not self.can_handle(entry):
            return
        
        with self._lock:
            self.entries.append(entry)
            
            # Limit memory usage
            if len(self.entries) > self.max_entries:
                self.entries = self.entries[-self.max_entries:]
    
    def get_entries(self, level: LogLevel = None, 
                   logger_name: str = None, limit: int = None) -> List[LogEntry]:
        """Get stored log entries with optional filtering"""
        with self._lock:
            entries = self.entries.copy()
        
        # Apply filters
        if level:
            entries = [e for e in entries if e.level == level]
        
        if logger_name:
            entries = [e for e in entries if e.logger_name == logger_name]
        
        # Apply limit
        if limit:
            entries = entries[-limit:]
        
        return entries
    
    def clear(self):
        """Clear stored entries"""
        with self._lock:
            self.entries.clear()


class OmniLogger:
    """
    Main logger class for OmniCAD.
    
    Provides structured logging with multiple handlers and performance monitoring.
    """
    
    _global_handlers: List[LogHandler] = []
    _loggers: Dict[str, 'OmniLogger'] = {}
    _lock = threading.Lock()
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.handlers: List[LogHandler] = []
        
        # Performance tracking
        self.start_time = datetime.now()
        self.log_count = 0
        self.error_count = 0
        
        # Register this logger
        with self._lock:
            self._loggers[name] = self
    
    @classmethod
    def setup_global_handlers(cls, log_dir: str = "logs", 
                            console_level: LogLevel = LogLevel.INFO,
                            file_level: LogLevel = LogLevel.DEBUG):
        """Set up global log handlers"""
        with cls._lock:
            cls._global_handlers.clear()
            
            # Console handler
            console_handler = ConsoleHandler(level=console_level)
            cls._global_handlers.append(console_handler)
            
            # File handler
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"omnicad_{datetime.now().strftime('%Y%m%d')}.log")
                file_handler = FileHandler(log_file, level=file_level)
                cls._global_handlers.append(file_handler)
            
            # Memory handler for runtime inspection
            memory_handler = MemoryHandler()
            cls._global_handlers.append(memory_handler)
    
    @classmethod
    def get_memory_handler(cls) -> Optional[MemoryHandler]:
        """Get the global memory handler"""
        for handler in cls._global_handlers:
            if isinstance(handler, MemoryHandler):
                return handler
        return None
    
    def add_handler(self, handler: LogHandler):
        """Add a handler to this logger"""
        self.handlers.append(handler)
    
    def remove_handler(self, handler: LogHandler):
        """Remove a handler from this logger"""
        if handler in self.handlers:
            self.handlers.remove(handler)
    
    def _log(self, level: LogLevel, message: str, extra_data: Dict[str, Any] = None):
        """Internal logging method"""
        if not self.enabled:
            return
        
        # Get caller information
        frame = sys._getframe(2)
        module = frame.f_globals.get('__name__', 'unknown')
        function = frame.f_code.co_name
        line = frame.f_lineno
        
        # Create log entry
        entry = LogEntry(
            level=level,
            message=message,
            logger_name=self.name,
            module=module,
            function=function,
            line=line,
            extra_data=extra_data
        )
        
        # Update counters
        self.log_count += 1
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self.error_count += 1
        
        # Send to all handlers
        all_handlers = self.handlers + self._global_handlers
        
        for handler in all_handlers:
            try:
                handler.handle(entry)
            except Exception as e:
                # Avoid infinite recursion if logging fails
                print(f"Log handler error: {e}", file=sys.stderr)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(LogLevel.INFO, message, kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log(LogLevel.WARNING, message, kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log(LogLevel.ERROR, message, kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log(LogLevel.CRITICAL, message, kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        exc_info = sys.exc_info()
        if exc_info[0] is not None:
            tb = ''.join(traceback.format_exception(*exc_info))
            kwargs['traceback'] = tb
        
        self._log(LogLevel.ERROR, message, kwargs)
    
    def performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        kwargs.update({
            'operation': operation,
            'duration_ms': duration * 1000,
            'performance': True
        })
        self._log(LogLevel.INFO, f"Performance: {operation} took {duration:.3f}s", kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics"""
        uptime = datetime.now() - self.start_time
        
        return {
            'name': self.name,
            'enabled': self.enabled,
            'uptime_seconds': uptime.total_seconds(),
            'log_count': self.log_count,
            'error_count': self.error_count,
            'handlers': len(self.handlers)
        }
    
    @classmethod
    def get_all_loggers(cls) -> Dict[str, 'OmniLogger']:
        """Get all registered loggers"""
        with cls._lock:
            return cls._loggers.copy()
    
    @classmethod
    def shutdown(cls):
        """Shutdown all loggers and handlers"""
        with cls._lock:
            for handler in cls._global_handlers:
                if hasattr(handler, 'close'):
                    handler.close()
            cls._global_handlers.clear()


# Initialize global logging
OmniLogger.setup_global_handlers()