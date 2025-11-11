#!/usr/bin/env python3
"""
Standalone test script for the capabilities module.
This script tests the capabilities module without requiring the full easygraph package to be built.
"""

import sys
import os

# Test the capabilities module directly
def test_capabilities():
    """Test the capabilities module functionality."""
    print("="*80)
    print("Testing EasyGraph Capabilities Module")
    print("="*80)
    print()
    
    # Load the module directly
    capabilities_path = os.path.join(
        os.path.dirname(__file__), 
        'easygraph', 
        'capabilities.py'
    )
    
    with open(capabilities_path, 'r') as f:
        code = f.read()
    
    # Execute the module
    namespace = {}
    exec(code, namespace)
    
    # Get the functions
    show_capabilities = namespace['show_capabilities']
    get_capabilities_dict = namespace['get_capabilities_dict']
    能做什么 = namespace['能做什么']
    
    # Test 1: show_capabilities is callable
    print("✓ Test 1: show_capabilities is callable")
    assert callable(show_capabilities)
    
    # Test 2: get_capabilities_dict is callable
    print("✓ Test 2: get_capabilities_dict is callable")
    assert callable(get_capabilities_dict)
    
    # Test 3: Chinese alias exists
    print("✓ Test 3: Chinese alias '能做什么' exists")
    assert callable(能做什么)
    
    # Test 4: Chinese alias is the same as show_capabilities
    print("✓ Test 4: Chinese alias equals show_capabilities")
    assert 能做什么 == show_capabilities
    
    # Test 5: get_capabilities_dict returns a dict
    print("✓ Test 5: get_capabilities_dict returns a dictionary")
    caps = get_capabilities_dict()
    assert isinstance(caps, dict)
    
    # Test 6: Dictionary has expected categories
    print("✓ Test 6: Dictionary has expected categories")
    expected_categories = [
        'centrality', 'community_detection', 'structural_holes',
        'graph_embedding', 'hypergraph', 'gpu_acceleration'
    ]
    for category in expected_categories:
        assert category in caps, f"Missing category: {category}"
    
    # Test 7: Categories have content
    print("✓ Test 7: Categories contain items")
    for category in expected_categories:
        assert len(caps[category]) > 0, f"Empty category: {category}"
    
    # Test 8: show_capabilities runs without error
    print("✓ Test 8: show_capabilities runs without error")
    show_capabilities()
    
    print()
    print("="*80)
    print("All tests passed!")
    print("="*80)
    print()
    
    # Show sample capabilities
    print("Sample capabilities:")
    print(f"  Centrality measures: {caps['centrality'][:3]}")
    print(f"  Community detection: {caps['community_detection'][:3]}")
    print(f"  Graph embedding: {caps['graph_embedding']}")
    print()
    
    return True


if __name__ == '__main__':
    try:
        success = test_capabilities()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
