"""
Event System Implementation

Provides a centralized event system for loose coupling between modules.
All communication between components should go through this event system.
"""

from typing import Dict, List, Callable, Any, Optional
import threading
import uuid
from datetime import datetime

from .logger import OmniLogger


class EventListener:
    """Represents an event listener"""
    
    def __init__(self, event_type: str, callback: Callable, priority: int = 0, once: bool = False):
        self.id = str(uuid.uuid4())
        self.event_type = event_type
        self.callback = callback
        self.priority = priority
        self.once = once
        self.created = datetime.now()
        self.call_count = 0


class EventSystem:
    """
    Centralized event system for OmniCAD.
    
    Provides publish-subscribe pattern for loose coupling between modules.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.listeners: Dict[str, List[EventListener]] = {}
        self.logger = OmniLogger("EventSystem")
        self._event_history: List[Dict[str, Any]] = []
        self._max_history = 1000
        self._lock = threading.RLock()
        self._initialized = True
    
    def on(self, event_type: str, callback: Callable, priority: int = 0, once: bool = False) -> str:
        """
        Register an event listener.
        
        Args:
            event_type: Type of event to listen for
            callback: Function to call when event is emitted
            priority: Priority of listener (higher = called first)
            once: If True, listener is removed after first call
        
        Returns:
            Listener ID for removal
        """
        with self._lock:
            listener = EventListener(event_type, callback, priority, once)
            
            if event_type not in self.listeners:
                self.listeners[event_type] = []
            
            self.listeners[event_type].append(listener)
            
            # Sort by priority (descending)
            self.listeners[event_type].sort(key=lambda x: x.priority, reverse=True)
            
            self.logger.debug(f"Registered listener for '{event_type}' (ID: {listener.id})")
            return listener.id
    
    def off(self, event_type: str = None, listener_id: str = None, callback: Callable = None):
        """
        Remove event listener(s).
        
        Args:
            event_type: Event type to remove listeners from
            listener_id: Specific listener ID to remove
            callback: Specific callback function to remove
        """
        with self._lock:
            if listener_id:
                # Remove specific listener by ID
                for event_type_key, listeners in self.listeners.items():
                    self.listeners[event_type_key] = [
                        l for l in listeners if l.id != listener_id
                    ]
                    
            elif event_type and callback:
                # Remove specific callback for event type
                if event_type in self.listeners:
                    self.listeners[event_type] = [
                        l for l in self.listeners[event_type] if l.callback != callback
                    ]
                    
            elif event_type:
                # Remove all listeners for event type
                if event_type in self.listeners:
                    del self.listeners[event_type]
                    
            elif callback:
                # Remove callback from all event types
                for event_type_key in self.listeners:
                    self.listeners[event_type_key] = [
                        l for l in self.listeners[event_type_key] if l.callback != callback
                    ]
            
            self.logger.debug(f"Removed listener(s) for '{event_type}'")
    
    def emit(self, event_type: str, data: Any = None, source: str = None):
        """
        Emit an event to all registered listeners.
        
        Args:
            event_type: Type of event to emit
            data: Data to pass to listeners
            source: Source identifier for the event
        """
        with self._lock:
            timestamp = datetime.now()
            
            # Create event object
            event = {
                'type': event_type,
                'data': data,
                'source': source,
                'timestamp': timestamp.isoformat()
            }
            
            # Add to history
            self._add_to_history(event)
            
            # Get listeners for this event type
            listeners = self.listeners.get(event_type, [])
            
            if not listeners:
                self.logger.debug(f"No listeners for event '{event_type}'")
                return
            
            self.logger.debug(f"Emitting '{event_type}' to {len(listeners)} listener(s)")
            
            # Call listeners in priority order
            listeners_to_remove = []
            
            for listener in listeners:
                try:
                    # Call the listener
                    listener.callback(data)
                    listener.call_count += 1
                    
                    # Mark for removal if 'once' listener
                    if listener.once:
                        listeners_to_remove.append(listener)
                        
                except Exception as e:
                    self.logger.error(f"Error in event listener for '{event_type}': {str(e)}")
            
            # Remove 'once' listeners
            for listener in listeners_to_remove:
                self.listeners[event_type].remove(listener)
    
    def emit_async(self, event_type: str, data: Any = None, source: str = None):
        """
        Emit an event asynchronously (non-blocking).
        
        Args:
            event_type: Type of event to emit
            data: Data to pass to listeners
            source: Source identifier for the event
        """
        import threading
        
        def emit_thread():
            self.emit(event_type, data, source)
        
        thread = threading.Thread(target=emit_thread, daemon=True)
        thread.start()
    
    def once(self, event_type: str, callback: Callable, priority: int = 0) -> str:
        """
        Register a one-time event listener.
        
        Args:
            event_type: Type of event to listen for
            callback: Function to call when event is emitted
            priority: Priority of listener
        
        Returns:
            Listener ID
        """
        return self.on(event_type, callback, priority, once=True)
    
    def get_listeners(self, event_type: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get information about registered listeners.
        
        Args:
            event_type: Specific event type to get listeners for
        
        Returns:
            Dictionary of listener information
        """
        result = {}
        
        event_types = [event_type] if event_type else self.listeners.keys()
        
        for et in event_types:
            if et in self.listeners:
                result[et] = []
                for listener in self.listeners[et]:
                    result[et].append({
                        'id': listener.id,
                        'priority': listener.priority,
                        'once': listener.once,
                        'created': listener.created.isoformat(),
                        'call_count': listener.call_count
                    })
        
        return result
    
    def get_event_history(self, event_type: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get event history.
        
        Args:
            event_type: Filter by event type
            limit: Maximum number of events to return
        
        Returns:
            List of historical events
        """
        history = self._event_history
        
        if event_type:
            history = [e for e in history if e['type'] == event_type]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def clear_history(self):
        """Clear event history"""
        with self._lock:
            self._event_history.clear()
            self.logger.debug("Event history cleared")
    
    def clear_listeners(self, event_type: str = None):
        """Clear all listeners for an event type or all listeners"""
        with self._lock:
            if event_type:
                if event_type in self.listeners:
                    del self.listeners[event_type]
                    self.logger.debug(f"Cleared all listeners for '{event_type}'")
            else:
                self.listeners.clear()
                self.logger.debug("Cleared all event listeners")
    
    def _add_to_history(self, event: Dict[str, Any]):
        """Add event to history"""
        self._event_history.append(event)
        
        # Limit history size
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event system statistics"""
        with self._lock:
            listener_count = sum(len(listeners) for listeners in self.listeners.values())
            
            return {
                'total_listeners': listener_count,
                'event_types': len(self.listeners),
                'history_size': len(self._event_history),
                'listeners_by_type': {
                    event_type: len(listeners) 
                    for event_type, listeners in self.listeners.items()
                }
            }