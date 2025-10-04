/**
 * Pyodide Loader for OmniCAD
 * 
 * Handles loading and initialization of the Python environment in the browser.
 */

class PyodideLoader {
    constructor() {
        this.pyodide = null;
        this.isLoaded = false;
        this.loadingProgress = 0;
        this.logger = new Logger('PyodideLoader');
        
        // Track loading stages
        this.loadingStages = [
            'Loading Pyodide runtime...',
            'Installing Python packages...',
            'Loading OmniCAD modules...',
            'Initializing geometry kernel...',
            'Setting up UI integration...',
            'Ready!'
        ];
        this.currentStage = 0;
    }
    
    async load() {
        try {
            this.logger.info('Starting Pyodide initialization');
            this.updateLoadingProgress(0, this.loadingStages[0]);
            
            // Load Pyodide with timeout
            this.pyodide = await Promise.race([
                loadPyodide({
                    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/",
                    packages: ["numpy"]
                }),
                new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('Pyodide loading timeout')), 30000)
                )
            ]);
            
            this.updateLoadingProgress(20, this.loadingStages[1]);
            
            // Install additional packages (optional)
            try {
                await this.installPackages();
            } catch (error) {
                this.logger.warning('Some packages failed to install, continuing:', error);
            }
            
            this.updateLoadingProgress(40, this.loadingStages[2]);
            
            // Load OmniCAD Python modules
            await this.loadOmniCADModules();
            
            this.updateLoadingProgress(60, this.loadingStages[3]);
            
            // Initialize the application
            await this.initializeApp();
            
            this.updateLoadingProgress(80, this.loadingStages[4]);
            
            // Set up Python-JavaScript bridge
            this.setupBridge();
            
            this.updateLoadingProgress(100, this.loadingStages[5]);
            
            this.isLoaded = true;
            this.logger.info('Pyodide initialization completed');
            
            return this.pyodide;
            
        } catch (error) {
            this.logger.error('Failed to load Pyodide:', error);
            // Continue without Python backend
            this.isLoaded = false;
            this.updateLoadingProgress(100, 'Ready (JavaScript only)');
            throw error;
        }
    }
    
    async installPackages() {
        // Install only essential packages to avoid loading issues
        const packages = [
            'sympy'
        ];
        
        for (const pkg of packages) {
            try {
                await Promise.race([
                    this.pyodide.loadPackage(pkg),
                    new Promise((_, reject) => 
                        setTimeout(() => reject(new Error(`Package ${pkg} timeout`)), 15000)
                    )
                ]);
                this.logger.debug(`Installed package: ${pkg}`);
            } catch (error) {
                this.logger.warning(`Failed to install package ${pkg}:`, error);
            }
        }
    }
    
    async loadOmniCADModules() {
        // Create simplified Python modules for web environment
        await this.pyodide.runPython(`
import sys
import os
from pathlib import Path

print("Creating OmniCAD Python modules...")

# Create minimal implementations for web environment
class OmniCADApp:
    def __init__(self):
        self.initialized = False
        self.modules = {}
        self.event_system = EventSystem()
        self.command_history = []
        self.undo_stack = []
        self.redo_stack = []
        
    async def initialize(self):
        self.initialized = True
        print("OmniCAD App initialized successfully")
        return self
        
    def get_module(self, name):
        if name not in self.modules:
            if name == 'geometry':
                self.modules[name] = GeometryKernel()
            elif name == 'sketch':
                self.modules[name] = SketchModule()
            else:
                self.modules[name] = Module(name)
        return self.modules[name]
        
    def execute_command(self, command_name, **params):
        print(f"Executing command: {command_name} with params: {params}")
        self.command_history.append((command_name, params))
        return f"Command {command_name} executed"
        
    def undo(self):
        if self.undo_stack:
            command = self.undo_stack.pop()
            self.redo_stack.append(command)
            print(f"Undoing: {command}")
            
    def redo(self):
        if self.redo_stack:
            command = self.redo_stack.pop()
            self.undo_stack.append(command)
            print(f"Redoing: {command}")

class EventSystem:
    def __init__(self):
        self.listeners = {}
        
    def on(self, event_type, callback):
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)
        
    def emit(self, event_type, data=None):
        if event_type in self.listeners:
            for callback in self.listeners[event_type]:
                callback(data)

class Module:
    def __init__(self, name):
        self.name = name
        
class GeometryKernel:
    def __init__(self):
        self.points = []
        self.lines = []
        self.circles = []
        
    def create_point(self, x, y, z=0):
        point = {'type': 'point', 'x': x, 'y': y, 'z': z, 'id': len(self.points)}
        self.points.append(point)
        print(f"Created point: ({x}, {y}, {z})")
        return point
        
    def create_line(self, start, end):
        line = {'type': 'line', 'start': start, 'end': end, 'id': len(self.lines)}
        self.lines.append(line)
        print(f"Created line from {start} to {end}")
        return line
        
    def create_circle(self, center, radius, normal=None):
        circle = {'type': 'circle', 'center': center, 'radius': radius, 'normal': normal, 'id': len(self.circles)}
        self.circles.append(circle)
        print(f"Created circle at {center} with radius {radius}")
        return circle
            
class SketchModule:
    def __init__(self):
        self.sketches = []
        
    def create_sketch(self, name="Sketch"):
        sketch = Sketch(name)
        self.sketches.append(sketch)
        return sketch
        
class Sketch:
    def __init__(self, name="Sketch"):
        self.name = name
        self.entities = []
        self.constraints = []
        
    def add_entity(self, entity):
        self.entities.append(entity)
        return entity
        
    def add_constraint(self, constraint):
        self.constraints.append(constraint)
        return constraint

# Make available globally
globals()['OmniCADApp'] = OmniCADApp
globals()['GeometryKernel'] = GeometryKernel
globals()['Sketch'] = Sketch
globals()['EventSystem'] = EventSystem

print("OmniCAD Python modules created successfully")
        `);
    }
    
    async initializeApp() {
        await this.pyodide.runPython(`
# Initialize the OmniCAD application
print("Initializing OmniCAD application...")
app = OmniCADApp()

# Initialize the app
import asyncio

# Create event loop if it doesn't exist
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Initialize app synchronously for web environment
app.initialized = True
print("OmniCAD application initialized successfully")

# Store reference for JavaScript access
import js
js.omnicad_app = app
print("OmniCAD application instance created and ready")
        `);
    }
    
    setupBridge() {
        // Create JavaScript-Python bridge
        window.omnicad = {
            // Core application reference
            app: this.pyodide.globals.get('app'),
            pyodide: this.pyodide,
            
            // Utility functions
            runPython: (code) => this.pyodide.runPython(code),
            
            // Geometry operations
            geometry: {
                createPoint: (x, y, z) => {
                    return this.pyodide.runPython(`
app.get_module('geometry').create_point(${x}, ${y}, ${z})
                    `);
                },
                
                createLine: (start, end) => {
                    return this.pyodide.runPython(`
app.get_module('geometry').create_line([${start.join(',')}], [${end.join(',')}])
                    `);
                },
                
                createCircle: (center, radius, normal = null) => {
                    const normalStr = normal ? `[${normal.join(',')}]` : 'None';
                    return this.pyodide.runPython(`
app.get_module('geometry').create_circle([${center.join(',')}], ${radius}, ${normalStr})
                    `);
                }
            },
            
            // Sketch operations
            sketch: {
                create: (name = 'Sketch') => {
                    return this.pyodide.runPython(`
from src.geometry.sketch import Sketch
sketch = Sketch("${name}")
sketch
                    `);
                }
            },
            
            // Command system
            commands: {
                execute: (commandName, params = {}) => {
                    const paramsStr = JSON.stringify(params);
                    return this.pyodide.runPython(`
import json
params = json.loads('${paramsStr}')
app.execute_command("${commandName}", **params)
                    `);
                },
                
                undo: () => {
                    return this.pyodide.runPython('app.undo()');
                },
                
                redo: () => {
                    return this.pyodide.runPython('app.redo()');
                }
            },
            
            // Event system
            events: {
                on: (eventType, callback) => {
                    // Set up event listener bridge
                    this.pyodide.runPython(`
def js_callback(data):
    import js
    js.callPythonCallback("${eventType}", data)

app.event_system.on("${eventType}", js_callback)
                    `);
                    
                    // Store JavaScript callback
                    if (!window.pythonEventCallbacks) {
                        window.pythonEventCallbacks = {};
                    }
                    window.pythonEventCallbacks[eventType] = callback;
                }
            }
        };
        
        // Global callback function for Python events
        window.callPythonCallback = (eventType, data) => {
            const callback = window.pythonEventCallbacks?.[eventType];
            if (callback) {
                callback(data);
            }
        };
        
        this.logger.info('Python-JavaScript bridge established');
    }
    
    updateLoadingProgress(progress, message) {
        this.loadingProgress = progress;
        
        // Update loading screen
        const progressBar = document.querySelector('.loading-progress');
        const loadingText = document.querySelector('.loading-text');
        
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }
        
        if (loadingText) {
            loadingText.textContent = message;
        }
        
        this.logger.debug(`Loading progress: ${progress}% - ${message}`);
    }
    
    getLoadingProgress() {
        return {
            progress: this.loadingProgress,
            stage: this.currentStage,
            message: this.loadingStages[this.currentStage] || 'Loading...',
            isComplete: this.isLoaded
        };
    }
}

// Simple logger for browser environment
class Logger {
    constructor(name) {
        this.name = name;
    }
    
    debug(message, ...args) {
        console.debug(`[${this.name}] ${message}`, ...args);
    }
    
    info(message, ...args) {
        console.info(`[${this.name}] ${message}`, ...args);
    }
    
    warning(message, ...args) {
        console.warn(`[${this.name}] ${message}`, ...args);
    }
    
    error(message, ...args) {
        console.error(`[${this.name}] ${message}`, ...args);
    }
}

// Export for use in main application
window.PyodideLoader = PyodideLoader;