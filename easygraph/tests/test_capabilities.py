"""
Tests for the capabilities module.
"""

import os
import sys


# Add parent directory to path for direct testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest

from io import StringIO


class TestCapabilities(unittest.TestCase):
    """Test cases for capabilities module."""

    def test_show_capabilities_import(self):
        """Test that show_capabilities can be imported."""
        from easygraph.capabilities import show_capabilities

        self.assertIsNotNone(show_capabilities)
        self.assertTrue(callable(show_capabilities))

    def test_show_capabilities_runs(self):
        """Test that show_capabilities runs without error."""
        from easygraph.capabilities import show_capabilities

        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            show_capabilities()
            output = captured_output.getvalue()

            # Check that output contains expected keywords
            self.assertIn("EasyGraph", output)
            self.assertIn("你能做什么", output)
            self.assertIn("CENTRALITY", output)
            self.assertIn("COMMUNITY", output)
            self.assertIn("STRUCTURAL HOLE", output)
            self.assertIn("pagerank", output)
            self.assertIn("louvain", output)

        finally:
            sys.stdout = old_stdout

    def test_get_capabilities_dict_import(self):
        """Test that get_capabilities_dict can be imported."""
        from easygraph.capabilities import get_capabilities_dict

        self.assertIsNotNone(get_capabilities_dict)
        self.assertTrue(callable(get_capabilities_dict))

    def test_get_capabilities_dict_structure(self):
        """Test the structure of the capabilities dictionary."""
        from easygraph.capabilities import get_capabilities_dict

        caps = get_capabilities_dict()

        # Check that it returns a dictionary
        self.assertIsInstance(caps, dict)

        # Check that expected categories exist
        expected_categories = [
            "centrality",
            "community_detection",
            "structural_holes",
            "graph_embedding",
            "hypergraph",
        ]

        for category in expected_categories:
            self.assertIn(category, caps, f"Category '{category}' missing")
            self.assertIsInstance(caps[category], list)
            self.assertTrue(len(caps[category]) > 0)

    def test_centrality_capabilities(self):
        """Test that centrality capabilities are correctly listed."""
        from easygraph.capabilities import get_capabilities_dict

        caps = get_capabilities_dict()
        centrality = caps["centrality"]

        # Check some expected centrality measures
        expected_measures = ["pagerank", "betweenness_centrality", "degree_centrality"]
        for measure in expected_measures:
            self.assertTrue(
                any(measure in item for item in centrality),
                f"Expected centrality measure '{measure}' not found",
            )

    def test_chinese_alias(self):
        """Test that the Chinese alias exists and works."""
        from easygraph.capabilities import 能做什么

        self.assertIsNotNone(能做什么)
        self.assertTrue(callable(能做什么))

        # Test that it's the same as show_capabilities
        from easygraph.capabilities import show_capabilities

        self.assertEqual(能做什么, show_capabilities)

    def test_all_exports(self):
        """Test that __all__ contains the expected exports."""
        from easygraph import capabilities

        self.assertTrue(hasattr(capabilities, "__all__"))
        expected_exports = ["show_capabilities", "get_capabilities_dict", "能做什么"]

        for export in expected_exports:
            self.assertIn(export, capabilities.__all__)


if __name__ == "__main__":
    unittest.main()
