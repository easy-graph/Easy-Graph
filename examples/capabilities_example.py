"""
Example: Using EasyGraph Capabilities Discovery Feature

This example demonstrates how to use the new capabilities discovery feature
to find out what EasyGraph can do (你能做什么？).
"""

# The following examples show how to use the capabilities feature
# once EasyGraph is installed.

# Note: These examples are for demonstration. Since the C++ extension
# may not be built in development environments, we show how to use
# the feature conceptually.


def example_usage():
    """
    Example usage of the capabilities discovery feature.
    """
    print("=" * 80)
    print("EasyGraph Capabilities Discovery - Example Usage")
    print("=" * 80)
    print()

    # Example 1: Display all capabilities
    print("Example 1: Show all capabilities")
    print("-" * 80)
    print(">>> import easygraph as eg")
    print(">>> eg.show_capabilities()")
    print()
    print("This will display a comprehensive overview of all EasyGraph features,")
    print("including centrality measures, community detection, structural holes,")
    print("graph embedding, hypergraph analysis, GPU acceleration, and more.")
    print()

    # Example 2: Get capabilities as a dictionary
    print("Example 2: Get capabilities as a dictionary")
    print("-" * 80)
    print(">>> import easygraph as eg")
    print(">>> caps = eg.get_capabilities_dict()")
    print(">>> print(caps.keys())")
    print()
    print("This returns a dictionary with categories like:")
    print("  - centrality")
    print("  - community_detection")
    print("  - structural_holes")
    print("  - graph_embedding")
    print("  - hypergraph")
    print("  - gpu_acceleration")
    print("  - and more...")
    print()

    # Example 3: Explore specific categories
    print("Example 3: Explore specific categories")
    print("-" * 80)
    print(">>> caps = eg.get_capabilities_dict()")
    print(">>> print('Centrality measures:', caps['centrality'])")
    print(">>> print('Community detection:', caps['community_detection'])")
    print()

    # Example 4: Chinese language support
    print("Example 4: Chinese language support (中文支持)")
    print("-" * 80)
    print(">>> import easygraph as eg")
    print(">>> eg.能做什么()")
    print()
    print("The Chinese alias '能做什么' (What can you do?) is available")
    print("for Chinese-speaking users.")
    print()

    # Example 5: Programmatic feature checking
    print("Example 5: Programmatic feature checking")
    print("-" * 80)
    print(">>> caps = eg.get_capabilities_dict()")
    print(">>> if 'pagerank' in str(caps['centrality']):")
    print(">>>     print('PageRank is available!')")
    print()

    print("=" * 80)
    print("For more information, visit: https://easy-graph.github.io/")
    print("=" * 80)


if __name__ == "__main__":
    example_usage()
