import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpp_modularity_test import TestModularity
from cpp_greedy_modularity_test import TestGreedyModularity
from cpp_enumerate_subgraph_test import (
    TestEnumerateSubgraph,
    TestRandomEnumerateSubgraph
)
from cpp_louvain_test import TestLouvainCommunities, TestLouvainCommunitiesSerial
from cpp_LPA_test import TestLPA
from cpp_ego_graph_test import TestEgoGraph, TestEgoGraphCSR
from cpp_localsearch_test import TestLocalsearch


def create_test_suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestModularity))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGreedyModularity))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEnumerateSubgraph))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRandomEnumerateSubgraph))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLouvainCommunities))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLouvainCommunitiesSerial))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLPA))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEgoGraph))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEgoGraphCSR))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLocalsearch))
    return suite


if __name__ == "__main__":
    print("=" * 70)
    print("Easy-Graph C++ Module Test Suite")
    print("=" * 70)
    print()

    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)

    sys.exit(0 if result.wasSuccessful() else 1)