# Capabilities Discovery Feature - Demonstration

## Overview
This document demonstrates the new capabilities discovery feature added to EasyGraph in response to the question "你能做什么？" (What can you do?).

## Feature Description
The capabilities module provides two main functions to help users discover what EasyGraph can do:

1. **`show_capabilities()`** - Displays a comprehensive, formatted overview
2. **`get_capabilities_dict()`** - Returns capabilities as a structured dictionary
3. **`能做什么()`** - Chinese language alias for `show_capabilities()`

## Usage Examples

### Example 1: Display All Capabilities
```python
import easygraph as eg
eg.show_capabilities()
```

### Example 2: Get Capabilities Programmatically
```python
import easygraph as eg
caps = eg.get_capabilities_dict()

# View available categories
print(caps.keys())
# Output: dict_keys(['graph_types', 'centrality', 'community_detection', 
#                    'structural_holes', 'components', 'basic_metrics', 
#                    'path_algorithms', 'core_decomposition', 'graph_embedding', 
#                    'graph_generation', 'hypergraph', 'gpu_acceleration', 
#                    'visualization', 'io_formats'])

# View specific capabilities
print(caps['centrality'])
# Output: ['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 
#          'pagerank', 'katz_centrality', 'ego_betweenness', 
#          'flow_betweenness', 'laplacian_centrality']
```

### Example 3: Chinese Language Support
```python
import easygraph as eg
eg.能做什么()  # Same as show_capabilities()
```

## Sample Output
When you call `eg.show_capabilities()`, you'll see:

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    EasyGraph Capabilities (你能做什么？)                   ║
║                         What Can EasyGraph Do?                           ║
╚══════════════════════════════════════════════════════════════════════════╝

EasyGraph is a comprehensive network analysis library. Here's what it can do:

📊 GRAPH CREATION & MANIPULATION
  • Create graphs: Graph(), DiGraph(), MultiGraph(), MultiDiGraph()
  • Add/remove nodes and edges
  • Graph conversion between different types
  • Support for various input formats (edge lists, adjacency matrices, etc.)

📈 CENTRALITY MEASURES
  • Degree centrality
  • Betweenness centrality
  • Closeness centrality
  • PageRank
  • Katz centrality
  • Ego betweenness
  • Flow betweenness
  • Laplacian centrality

🔍 COMMUNITY DETECTION
  • Louvain algorithm
  • Label Propagation Algorithm (LPA)
  • Modularity-based detection
  • Ego graph extraction
  • Motif detection

🕳️ STRUCTURAL HOLE ANALYSIS
  • HIS (Structural Hole Information Diffusion)
  • HAM (Hierarchical Affiliation Model)
  • MaxD (Maximum Degree)
  • AP_Greedy
  • Constraint metrics
  • Effective size
  • Various structural hole evaluation metrics

... [and much more]
```

## Benefits

### For New Users
- Quickly discover what EasyGraph can do
- No need to read extensive documentation to find available features
- Clear categorization makes it easy to find relevant algorithms

### For Experienced Users
- Quick reference for available algorithms
- Programmatic access via dictionary for automation
- Chinese language support for Chinese-speaking users

### For Integration
- Can be used in interactive environments (Jupyter, IPython)
- Useful for building discovery tools or documentation
- Helps with feature exploration during development

## Implementation Details

### Files Added
- `easygraph/capabilities.py` - Main module (304 lines)
- `easygraph/tests/test_capabilities.py` - Unit tests (120 lines)
- `test_capabilities_standalone.py` - Standalone test script (104 lines)
- `examples/capabilities_example.py` - Usage examples (85 lines)

### Files Modified
- `easygraph/__init__.py` - Added import and export of capabilities module (2 lines added)

### Code Quality
- ✅ Black formatted
- ✅ Isort applied
- ✅ Flake8 compliant
- ✅ All tests pass
- ✅ Backward compatible

## Categories Covered

The capabilities dictionary includes the following categories:

1. **graph_types** - Available graph types
2. **centrality** - Centrality measures
3. **community_detection** - Community detection algorithms
4. **structural_holes** - Structural hole analysis methods
5. **components** - Network component analysis
6. **basic_metrics** - Basic network metrics
7. **path_algorithms** - Path finding algorithms
8. **core_decomposition** - Core decomposition methods
9. **graph_embedding** - Graph embedding techniques
10. **graph_generation** - Graph generation methods
11. **hypergraph** - Hypergraph analysis capabilities
12. **gpu_acceleration** - GPU-accelerated functions
13. **visualization** - Visualization capabilities
14. **io_formats** - Supported I/O formats

## Future Enhancements

Potential future improvements:
- Add version information for each feature
- Include links to documentation for each algorithm
- Add performance characteristics (time/space complexity)
- Filter capabilities by category or search term
- Generate capability reports

## Conclusion

This feature directly addresses the question "你能做什么？" (What can you do?) by providing:
1. A comprehensive overview of EasyGraph's capabilities
2. Easy programmatic access to capability information
3. Chinese language support for a global audience
4. Clear documentation and examples

The implementation is minimal, non-invasive, and maintains full backward compatibility with existing code.
