/**
 * Main JavaScript Application for OmniCAD
 * 
 * Handles UI interactions, viewport management, and integration with Python backend.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

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

class OmniCADUI {
    constructor() {
        this.logger = new Logger('OmniCADUI');
        this.pyodideLoader = new window.PyodideLoader();
        
        // Application state
        this.currentMode = 'model';
        this.selectedFeatures = [];
        
        // UI components
        this.viewport = null;
        this.featureTree = null;
        this.propertyPanel = null;
        
        // Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        
        // Event handlers
        this.eventHandlers = new Map();
        
        this.logger.info('OmniCAD UI initialized');
    }
    
    async initialize() {
        try {
            this.logger.info('Starting OmniCAD application initialization');
            
            // Show loading screen
            this.showLoadingScreen();
            
            // Load Python environment
            try {
                await this.pyodideLoader.load();
                this.logger.info('Python environment loaded successfully');
            } catch (error) {
                this.logger.warning('Python environment failed to load, continuing with JavaScript only:', error);
                // Continue without Python backend
            }
            
            // Initialize UI components
            this.initializeUI();
            
            // Initialize 3D viewport
            this.initializeViewport();
            
            // Set up event handlers
            this.setupEventHandlers();
            
            // Hide loading screen and show main app
            this.hideLoadingScreen();
            
            this.logger.info('OmniCAD application ready');
            
        } catch (error) {
            this.logger.error('Failed to initialize OmniCAD:', error);
            this.showError('Failed to initialize OmniCAD', error.message);
        }
    }
    
    showLoadingScreen() {
        const loadingScreen = document.getElementById('loading-screen');
        const mainApp = document.getElementById('main-app');
        
        if (loadingScreen) loadingScreen.style.display = 'flex';
        if (mainApp) mainApp.style.display = 'none';
    }
    
    hideLoadingScreen() {
        const loadingScreen = document.getElementById('loading-screen');
        const mainApp = document.getElementById('main-app');
        
        if (loadingScreen) {
            loadingScreen.style.display = 'none';
        }
        if (mainApp) {
            mainApp.style.display = 'flex';
            // Force a layout recalculation
            this.forceLayoutUpdate();
        }
    }
    
    initializeUI() {
        this.logger.debug('Initializing UI components');
        
        // Initialize feature tree
        this.initializeFeatureTree();
        
        // Initialize property panel
        this.initializePropertyPanel();
        
        // Initialize console
        this.initializeConsole();
        
        // Set initial mode
        this.setMode(this.currentMode);
    }
    
    initializeFeatureTree() {
        const treeContainer = document.getElementById('feature-tree');
        if (!treeContainer) return;
        
        // Create root tree structure
        const treeHTML = `
            <div class="tree-node expanded" data-node-type="document">
                <div class="tree-node-header">
                    <span class="tree-expand-icon">▼</span>
                    <span class="tree-node-icon">📄</span>
                    <span class="tree-node-text">Document</span>
                </div>
                <div class="tree-node-children">
                    <div class="tree-node" data-node-type="origin">
                        <div class="tree-node-header">
                            <span class="tree-node-icon">🎯</span>
                            <span class="tree-node-text">Origin</span>
                        </div>
                    </div>
                    <div class="tree-node" data-node-type="features">
                        <div class="tree-node-header">
                            <span class="tree-expand-icon">▶</span>
                            <span class="tree-node-icon">🔧</span>
                            <span class="tree-node-text">Features</span>
                        </div>
                        <div class="tree-node-children" style="display: none;"></div>
                    </div>
                    <div class="tree-node" data-node-type="sketches">
                        <div class="tree-node-header">
                            <span class="tree-expand-icon">▶</span>
                            <span class="tree-node-icon">✏️</span>
                            <span class="tree-node-text">Sketches</span>
                        </div>
                        <div class="tree-node-children" style="display: none;"></div>
                    </div>
                    <div class="tree-node" data-node-type="materials">
                        <div class="tree-node-header">
                            <span class="tree-expand-icon">▶</span>
                            <span class="tree-node-icon">🎨</span>
                            <span class="tree-node-text">Materials</span>
                        </div>
                        <div class="tree-node-children" style="display: none;"></div>
                    </div>
                </div>
            </div>
        `;
        
        treeContainer.innerHTML = treeHTML;
        
        // Add tree interaction handlers
        this.setupTreeInteractions();
    }
    
    setupTreeInteractions() {
        const treeContainer = document.getElementById('feature-tree');
        if (!treeContainer) return;
        
        // Handle tree node clicks
        treeContainer.addEventListener('click', (event) => {
            const nodeHeader = event.target.closest('.tree-node-header');
            if (!nodeHeader) return;
            
            const treeNode = nodeHeader.closest('.tree-node');
            const expandIcon = nodeHeader.querySelector('.tree-expand-icon');
            const children = treeNode.querySelector('.tree-node-children');
            
            // Handle expand/collapse
            if (event.target === expandIcon && children) {
                const isExpanded = treeNode.classList.contains('expanded');
                
                if (isExpanded) {
                    treeNode.classList.remove('expanded');
                    expandIcon.textContent = '▶';
                    children.style.display = 'none';
                } else {
                    treeNode.classList.add('expanded');
                    expandIcon.textContent = '▼';
                    children.style.display = 'block';
                }
                
                return;
            }
            
            // Handle node selection
            this.selectTreeNode(treeNode);
        });
    }
    
    selectTreeNode(treeNode) {
        // Remove previous selection
        const previousSelected = document.querySelector('.tree-node.selected');
        if (previousSelected) {
            previousSelected.classList.remove('selected');
        }
        
        // Add selection to new node
        treeNode.classList.add('selected');
        
        // Update property panel
        const nodeType = treeNode.dataset.nodeType;
        const nodeText = treeNode.querySelector('.tree-node-text').textContent;
        
        this.updatePropertyPanel(nodeType, nodeText);
        
        this.logger.debug(`Selected tree node: ${nodeType} - ${nodeText}`);
    }
    
    initializePropertyPanel() {
        const propertyEditor = document.getElementById('property-editor');
        if (!propertyEditor) return;
        
        propertyEditor.innerHTML = `
            <div class="property-section">
                <h4>Selection</h4>
                <p>No item selected</p>
            </div>
        `;
    }
    
    updatePropertyPanel(nodeType, nodeName) {
        const propertyEditor = document.getElementById('property-editor');
        if (!propertyEditor) return;
        
        let propertiesHTML = '';
        
        switch (nodeType) {
            case 'document':
                propertiesHTML = `
                    <div class="property-section">
                        <h4>Document Properties</h4>
                        <div class="property-row">
                            <label>Name:</label>
                            <input type="text" value="${nodeName}" />
                        </div>
                        <div class="property-row">
                            <label>Units:</label>
                            <select>
                                <option value="mm">Millimeters</option>
                                <option value="cm">Centimeters</option>
                                <option value="m">Meters</option>
                                <option value="in">Inches</option>
                                <option value="ft">Feet</option>
                            </select>
                        </div>
                        <div class="property-row">
                            <label>Precision:</label>
                            <input type="number" value="0.01" step="0.001" />
                        </div>
                    </div>
                `;
                break;
                
            case 'origin':
                propertiesHTML = `
                    <div class="property-section">
                        <h4>Origin</h4>
                        <div class="property-row">
                            <label>X:</label>
                            <input type="number" value="0" disabled />
                        </div>
                        <div class="property-row">
                            <label>Y:</label>
                            <input type="number" value="0" disabled />
                        </div>
                        <div class="property-row">
                            <label>Z:</label>
                            <input type="number" value="0" disabled />
                        </div>
                    </div>
                `;
                break;
                
            default:
                propertiesHTML = `
                    <div class="property-section">
                        <h4>${nodeName}</h4>
                        <p>Properties will be displayed here when features are selected.</p>
                    </div>
                `;
        }
        
        propertyEditor.innerHTML = propertiesHTML;
    }
    
    initializeConsole() {
        const consoleOutput = document.getElementById('console-output');
        if (!consoleOutput) return;
        
        // Add initial welcome message
        this.addConsoleMessage('OmniCAD v0.1.0 initialized successfully');
        this.addConsoleMessage('Ready for CAD/CAM operations');
        
        // Set up console tab switching
        const consoleTabs = document.querySelectorAll('.console-tab');
        consoleTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                this.switchConsoleTab(tab.dataset.tab);
            });
        });
    }
    
    addConsoleMessage(message, type = 'info') {
        const consoleOutput = document.getElementById('console-output');
        if (!consoleOutput) return;
        
        const timestamp = new Date().toLocaleTimeString();
        const messageElement = document.createElement('div');
        messageElement.className = `console-message console-${type}`;
        messageElement.innerHTML = `<span class="console-timestamp">[${timestamp}]</span> ${message}`;
        
        consoleOutput.appendChild(messageElement);
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }
    
    switchConsoleTab(tabName) {
        // Update tab selection
        const tabs = document.querySelectorAll('.console-tab');
        tabs.forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabName);
        });
        
        // Update content visibility
        const contents = document.querySelectorAll('.console-content');
        contents.forEach(content => {
            content.style.display = content.id === `console-${tabName}` ? 'block' : 'none';
        });
    }
    
    initializeViewport() {
        this.logger.debug('Initializing 3D viewport');
        
        const viewportContainer = document.getElementById('viewport-main');
        if (!viewportContainer) {
            this.logger.error('Viewport container not found');
            return;
        }
        
        // Wait for container to have dimensions
        const initWhenReady = () => {
            const width = viewportContainer.clientWidth || 800;
            const height = viewportContainer.clientHeight || 600;
            
            if (width === 0 || height === 0) {
                setTimeout(initWhenReady, 50);
                return;
            }
            
            // Initialize Three.js
            this.scene = new THREE.Scene();
            this.scene.background = new THREE.Color(0xf0f0f0);
            
            // Camera
            const aspect = width / height;
            this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
            this.camera.position.set(10, 10, 10);
            this.camera.lookAt(0, 0, 0);
            
            // Renderer
            this.renderer = new THREE.WebGLRenderer({ antialias: true });
            this.renderer.setSize(width, height);
            this.renderer.shadowMap.enabled = true;
            this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            
            // Clear any existing content
            viewportContainer.innerHTML = '';
            
            // Add renderer to container
            viewportContainer.appendChild(this.renderer.domElement);
            
            // Controls
            this.controls = new OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.1;
            
            // Lighting
            this.setupLighting();
            
            // Grid
            this.setupGrid();
            
            // Start render loop
            this.startRenderLoop();
            
            // Handle resize
            this.setupViewportResize();
            
            this.logger.debug('3D viewport initialized successfully');
        };
        
        initWhenReady();
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 20, 10);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);
        
        // Point light for fill
        const pointLight = new THREE.PointLight(0xffffff, 0.3);
        pointLight.position.set(-10, -10, 5);
        this.scene.add(pointLight);
    }
    
    setupGrid() {
        // Grid
        const grid = new THREE.GridHelper(20, 20, 0x888888, 0xcccccc);
        this.scene.add(grid);
        
        // Axes helper
        const axesHelper = new THREE.AxesHelper(5);
        this.scene.add(axesHelper);
    }
    
    startRenderLoop() {
        const animate = () => {
            requestAnimationFrame(animate);
            
            this.controls.update();
            this.renderer.render(this.scene, this.camera);
        };
        
        animate();
    }
    
    setupViewportResize() {
        const resizeObserver = new ResizeObserver(entries => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                
                if (width > 0 && height > 0) {
                    this.camera.aspect = width / height;
                    this.camera.updateProjectionMatrix();
                    this.renderer.setSize(width, height);
                }
            }
        });
        
        const viewportContainer = document.getElementById('viewport-main');
        if (viewportContainer) {
            resizeObserver.observe(viewportContainer);
        }
    }
    
    forceLayoutUpdate() {
        // Force layout recalculation
        setTimeout(() => {
            if (this.renderer && this.camera) {
                const viewportContainer = document.getElementById('viewport-main');
                if (viewportContainer) {
                    const width = viewportContainer.clientWidth;
                    const height = viewportContainer.clientHeight;
                    
                    if (width > 0 && height > 0) {
                        this.camera.aspect = width / height;
                        this.camera.updateProjectionMatrix();
                        this.renderer.setSize(width, height);
                    }
                }
            }
        }, 100);
    }
    
    setupEventHandlers() {
        this.logger.debug('Setting up event handlers');
        
        // Mode tabs
        const modeTabs = document.querySelectorAll('.mode-tabs .tab');
        modeTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                this.setMode(tab.dataset.mode);
            });
        });
        
        // File operations
        this.setupFileHandlers();
        
        // View controls
        this.setupViewControls();
        
        // Ribbon commands
        this.setupRibbonHandlers();
        
        // Keyboard shortcuts
        this.setupKeyboardHandlers();
    }
    
    setupFileHandlers() {
        const newBtn = document.getElementById('new-project');
        const openBtn = document.getElementById('open-project');
        const saveBtn = document.getElementById('save-project');
        const fileInput = document.getElementById('file-input');
        
        if (newBtn) {
            newBtn.addEventListener('click', () => this.newProject());
        }
        
        if (openBtn) {
            openBtn.addEventListener('click', () => fileInput?.click());
        }
        
        if (saveBtn) {
            saveBtn.addEventListener('click', () => this.saveProject());
        }
        
        if (fileInput) {
            fileInput.addEventListener('change', (event) => {
                const file = event.target.files[0];
                if (file) {
                    this.openProject(file);
                }
            });
        }
    }
    
    setupViewControls() {
        const viewButtons = {
            'view-iso': () => this.setViewIsometric(),
            'view-front': () => this.setViewFront(),
            'view-top': () => this.setViewTop(),
            'view-right': () => this.setViewRight(),
            'zoom-fit': () => this.zoomToFit()
        };
        
        Object.entries(viewButtons).forEach(([id, handler]) => {
            const button = document.getElementById(id);
            if (button) {
                button.addEventListener('click', handler);
            }
        });
    }
    
    setupRibbonHandlers() {
        const ribbonButtons = document.querySelectorAll('.ribbon-btn');
        ribbonButtons.forEach(button => {
            button.addEventListener('click', () => {
                const command = button.dataset.command;
                if (command) {
                    this.executeCommand(command);
                }
            });
        });
    }
    
    setupKeyboardHandlers() {
        document.addEventListener('keydown', (event) => {
            // Handle keyboard shortcuts
            if (event.ctrlKey || event.metaKey) {
                switch (event.key) {
                    case 'n':
                        event.preventDefault();
                        this.newProject();
                        break;
                    case 'o':
                        event.preventDefault();
                        document.getElementById('file-input')?.click();
                        break;
                    case 's':
                        event.preventDefault();
                        this.saveProject();
                        break;
                    case 'z':
                        event.preventDefault();
                        if (event.shiftKey) {
                            this.redo();
                        } else {
                            this.undo();
                        }
                        break;
                }
            }
        });
    }
    
    setMode(mode) {
        if (this.currentMode === mode) return;
        
        this.currentMode = mode;
        
        // Update mode tabs
        const tabs = document.querySelectorAll('.mode-tabs .tab');
        tabs.forEach(tab => {
            tab.classList.toggle('active', tab.dataset.mode === mode);
        });
        
        // Update ribbon
        const ribbonTabs = document.querySelectorAll('.ribbon-tab');
        ribbonTabs.forEach(tab => {
            tab.style.display = tab.dataset.mode === mode ? 'flex' : 'none';
        });
        
        // Update status bar
        const statusMode = document.querySelector('.status-mode');
        if (statusMode) {
            statusMode.textContent = `${mode.charAt(0).toUpperCase() + mode.slice(1)} Mode`;
        }
        
        this.logger.debug(`Mode changed to: ${mode}`);
        this.addConsoleMessage(`Switched to ${mode} mode`);
    }
    
    // View control methods
    setViewIsometric() {
        this.camera.position.set(10, 10, 10);
        this.camera.lookAt(0, 0, 0);
        this.controls.update();
    }
    
    setViewFront() {
        this.camera.position.set(0, 0, 15);
        this.camera.lookAt(0, 0, 0);
        this.controls.update();
    }
    
    setViewTop() {
        this.camera.position.set(0, 15, 0);
        this.camera.lookAt(0, 0, 0);
        this.controls.update();
    }
    
    setViewRight() {
        this.camera.position.set(15, 0, 0);
        this.camera.lookAt(0, 0, 0);
        this.controls.update();
    }
    
    zoomToFit() {
        // Implementation would calculate bounding box and adjust camera
        this.controls.reset();
    }
    
    // Command execution
    executeCommand(command) {
        this.logger.debug(`Executing command: ${command}`);
        this.addConsoleMessage(`Executing: ${command}`);
        
        try {
            // Use Python command system if available
            if (window.omnicad && window.omnicad.commands) {
                window.omnicad.commands.execute(command);
            } else {
                // Fallback to JavaScript implementations
                this.executeJSCommand(command);
            }
        } catch (error) {
            this.logger.error(`Command execution failed: ${command}`, error);
            this.addConsoleMessage(`Error executing ${command}: ${error.message}`, 'error');
        }
    }
    
    executeJSCommand(command) {
        switch (command) {
            case 'new-sketch':
                this.createNewSketch();
                break;
            case 'extrude':
                this.createExtrude();
                break;
            // Add more command implementations
            default:
                this.addConsoleMessage(`Command not implemented: ${command}`, 'warning');
        }
    }
    
    // File operations
    newProject() {
        this.logger.debug('Creating new project');
        this.addConsoleMessage('Creating new project...');
        
        // Clear scene
        while (this.scene.children.length > 0) {
            this.scene.remove(this.scene.children[0]);
        }
        
        // Re-add basic elements
        this.setupLighting();
        this.setupGrid();
        
        // Reset feature tree
        this.initializeFeatureTree();
        
        this.addConsoleMessage('New project created');
    }
    
    async openProject(file) {
        this.logger.debug(`Opening project: ${file.name}`);
        this.addConsoleMessage(`Opening project: ${file.name}`);
        
        try {
            const text = await file.text();
            const projectData = JSON.parse(text);
            
            // Load project data using Python backend if available
            if (window.omnicad && window.omnicad.app) {
                // Implementation would go here
            }
            
            this.addConsoleMessage(`Project opened: ${file.name}`);
            
        } catch (error) {
            this.logger.error('Failed to open project:', error);
            this.addConsoleMessage(`Error opening project: ${error.message}`, 'error');
        }
    }
    
    saveProject() {
        this.logger.debug('Saving project');
        this.addConsoleMessage('Saving project...');
        
        try {
            // Get project data from Python backend if available
            let projectData = {
                name: 'Untitled Project',
                version: '0.1.0',
                created: new Date().toISOString(),
                data: {}
            };
            
            if (window.omnicad && window.omnicad.app) {
                // Implementation would get data from Python
            }
            
            // Download as file
            const blob = new Blob([JSON.stringify(projectData, null, 2)], {
                type: 'application/json'
            });
            
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${projectData.name}.omniproject`;
            a.click();
            
            URL.revokeObjectURL(url);
            
            this.addConsoleMessage('Project saved');
            
        } catch (error) {
            this.logger.error('Failed to save project:', error);
            this.addConsoleMessage(`Error saving project: ${error.message}`, 'error');
        }
    }
    
    // Undo/Redo
    undo() {
        if (window.omnicad && window.omnicad.commands) {
            window.omnicad.commands.undo();
            this.addConsoleMessage('Undo');
        }
    }
    
    redo() {
        if (window.omnicad && window.omnicad.commands) {
            window.omnicad.commands.redo();
            this.addConsoleMessage('Redo');
        }
    }
    
    // Stub implementations for development
    createNewSketch() {
        this.addConsoleMessage('Creating new sketch...');
        // Implementation would create a sketch object
    }
    
    createExtrude() {
        this.addConsoleMessage('Creating extrude feature...');
        // Implementation would create an extrude feature
    }
    
    showError(title, message) {
        // Simple error display - would be replaced with proper modal
        alert(`${title}: ${message}`);
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.omnicadUI = new OmniCADUI();
    window.omnicadUI.initialize();
});

// Export for global access
window.OmniCADUI = OmniCADUI;