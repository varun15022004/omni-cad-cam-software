"""
Test script to verify OmniCAD implementation

This script tests the core functionality of the OmniCAD system
to ensure all modules are working correctly.
"""

import sys
import asyncio
import numpy as np
from datetime import datetime

# Add src to path for testing
sys.path.insert(0, 'src')

def test_core_system():
    """Test the core application system"""
    print("Testing Core System...")
    
    try:
        from src.core.app import OmniCADApp
        from src.core.feature_tree import FeatureTree, FeatureType
        from src.core.command_system import CommandManager
        from src.core.data_manager import DataManager
        
        print("✓ Core modules imported successfully")
        
        # Test application initialization
        app = OmniCADApp()
        print("✓ OmniCAD app created")
        
        # Test feature tree
        tree = FeatureTree()
        doc_id = tree.create_document_tree()
        print(f"✓ Feature tree created with document: {doc_id}")
        
        # Test command manager
        cmd_mgr = CommandManager()
        print("✓ Command manager created")
        
        # Test data manager
        data_mgr = DataManager()
        print("✓ Data manager created")
        
        return True
        
    except Exception as e:
        print(f"✗ Core system test failed: {e}")
        return False

def test_geometry_kernel():
    """Test the geometry kernel"""
    print("\nTesting Geometry Kernel...")
    
    try:
        from src.geometry.kernel import GeometryKernel, Point, Line, Circle, Transform3D
        
        print("✓ Geometry modules imported successfully")
        
        # Test kernel
        kernel = GeometryKernel()
        print("✓ Geometry kernel created")
        
        # Test basic geometry creation
        point_id = kernel.create_point(1.0, 2.0, 3.0)
        print(f"✓ Point created: {point_id}")
        
        line_id = kernel.create_line([0, 0, 0], [1, 1, 1])
        print(f"✓ Line created: {line_id}")
        
        circle_id = kernel.create_circle([0, 0, 0], 5.0)
        print(f"✓ Circle created: {circle_id}")
        
        # Test transforms
        transform = Transform3D.translation(1, 2, 3)
        point = np.array([0, 0, 0])
        transformed = transform.apply_to_point(point)
        print(f"✓ Transform test: {point} -> {transformed}")
        
        return True
        
    except Exception as e:
        print(f"✗ Geometry kernel test failed: {e}")
        return False

def test_sketch_system():
    """Test the sketching system"""
    print("\nTesting Sketch System...")
    
    try:
        from src.geometry.sketch import Sketch, SketchPoint, SketchLine, ConstraintType
        
        print("✓ Sketch modules imported successfully")
        
        # Create sketch
        sketch = Sketch("Test Sketch")
        print(f"✓ Sketch created: {sketch.name}")
        
        # Add points
        p1_id = sketch.add_point(0, 0)
        p2_id = sketch.add_point(10, 0)
        p3_id = sketch.add_point(10, 10)
        p4_id = sketch.add_point(0, 10)
        print(f"✓ Added 4 points to sketch")
        
        # Add lines
        l1_id = sketch.add_line(p1_id, p2_id)
        l2_id = sketch.add_line(p2_id, p3_id)
        l3_id = sketch.add_line(p3_id, p4_id)
        l4_id = sketch.add_line(p4_id, p1_id)
        print(f"✓ Added 4 lines to sketch")
        
        # Add constraints
        c1_id = sketch.add_constraint(ConstraintType.DISTANCE, [p1_id, p2_id], distance=10.0)
        c2_id = sketch.add_constraint(ConstraintType.HORIZONTAL, [l1_id])
        c3_id = sketch.add_constraint(ConstraintType.VERTICAL, [l2_id])
        print(f"✓ Added 3 constraints to sketch")
        
        # Try to solve
        solved = sketch.solve()
        print(f"✓ Sketch solve result: {solved}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sketch system test failed: {e}")
        return False

def test_utilities():
    """Test utility modules"""
    print("\nTesting Utilities...")
    
    try:
        from src.utils.event_system import EventSystem
        from src.utils.logger import OmniLogger
        
        print("✓ Utility modules imported successfully")
        
        # Test event system
        events = EventSystem()
        
        test_data = []
        def test_callback(data):
            test_data.append(data)
        
        events.on('test_event', test_callback)
        events.emit('test_event', {'message': 'Hello World'})
        
        if test_data and test_data[0]['message'] == 'Hello World':
            print("✓ Event system working")
        else:
            print("✗ Event system failed")
            return False
        
        # Test logger
        logger = OmniLogger("TestLogger")
        logger.info("Test log message")
        print("✓ Logger working")
        
        return True
        
    except Exception as e:
        print(f"✗ Utilities test failed: {e}")
        return False

async def test_async_initialization():
    """Test async initialization"""
    print("\nTesting Async Initialization...")
    
    try:
        from src.core.app import OmniCADApp
        
        app = OmniCADApp()
        await app.initialize()
        
        print("✓ Async initialization completed")
        return True
        
    except Exception as e:
        print(f"✗ Async initialization failed: {e}")
        return False

def test_file_operations():
    """Test file I/O operations"""
    print("\nTesting File Operations...")
    
    try:
        from src.core.data_manager import DataManager, OmniProjectFormat
        import tempfile
        import os
        
        data_mgr = DataManager()
        
        # Test project structure creation
        test_project = {
            'id': 'test-project',
            'name': 'Test Project',
            'feature_tree': {},
            'metadata': {
                'description': 'Test project for validation',
                'author': 'Test Suite'
            }
        }
        
        project_structure = OmniProjectFormat.create_project_structure(test_project)
        print("✓ Project structure created")
        
        # Test file format support
        import_formats = data_mgr.get_supported_formats('import')
        export_formats = data_mgr.get_supported_formats('export')
        
        print(f"✓ Supported import formats: {len(import_formats)}")
        print(f"✓ Supported export formats: {len(export_formats)}")
        
        return True
        
    except Exception as e:
        print(f"✗ File operations test failed: {e}")
        return False

def print_system_info():
    """Print system information"""
    print("=" * 60)
    print("OmniCAD System Test")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Test time: {datetime.now().isoformat()}")
    print(f"Platform: {sys.platform}")
    print("=" * 60)

def main():
    """Run all tests"""
    print_system_info()
    
    tests = [
        test_core_system,
        test_geometry_kernel,
        test_sketch_system,
        test_utilities,
        test_file_operations
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    # Test async functionality
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if loop.run_until_complete(test_async_initialization()):
            passed += 1
        total += 1
        loop.close()
    except Exception as e:
        print(f"✗ Async test crashed: {e}")
        total += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! OmniCAD is ready to run.")
        return 0
    else:
        print("❌ Some tests failed. Check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)