"""Offline regression tests for optional dependencies without importing EasyGraph."""

import builtins
import importlib.util
import sys
import types
import unittest
import warnings

from pathlib import Path
from unittest.mock import patch


OPTIONAL_PATH = Path(__file__).resolve().parents[2] / "easygraph" / "_optional.py"
INSTALL_URL = "https://pytorch.org/get-started/locally/"


def load_optional(torch_module=None, torch_error=None):
    """Load only the helper, with the Torch import fully controlled by the test."""
    spec = importlib.util.spec_from_file_location(
        "optional_dependency_under_test", OPTIONAL_PATH
    )
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__
    torch_imports = []

    def controlled_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "easygraph" or name.startswith("easygraph."):
            raise AssertionError("This test must not import the EasyGraph package")
        if name == "torch":
            torch_imports.append(name)
            if torch_error is not None:
                raise torch_error
            if torch_module is None:
                raise ModuleNotFoundError("No module named 'torch'", name="torch")
            return torch_module
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=controlled_import):
        spec.loader.exec_module(module)
    if torch_imports != ["torch"]:
        raise AssertionError("Expected exactly one controlled Torch import")
    return module


class OptionalDependencyTests(unittest.TestCase):
    def test_missing_torch_warns_with_current_python_and_official_install_link(self):
        helper = load_optional()
        self.assertFalse(helper.HAS_TORCH)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            helper.warn_missing_torch()
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, UserWarning)
        message = str(caught[0].message)
        self.assertIn(INSTALL_URL, message)
        self.assertIn("Python {}.{}".format(*sys.version_info[:2]), message)
        self.assertIn("Core graph algorithms", message)
        self.assertIn("C++ backend remain available", message)
        # The warning should point to the caller, where the user imported EasyGraph.
        self.assertEqual(Path(caught[0].filename).resolve(), Path(__file__).resolve())

    def test_require_torch_explains_feature_and_installation_when_missing(self):
        helper = load_optional()
        with self.assertRaises(ImportError) as caught:
            helper.require_torch("Hypergraph message passing")
        message = str(caught.exception)
        self.assertIn("Hypergraph message passing requires PyTorch", message)
        self.assertIn(INSTALL_URL, message)
        self.assertIn("Python version", message)
        self.assertIn("restart Python", message)

    def test_available_torch_is_returned_without_warning(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.__version__ = "test-only"
        helper = load_optional(torch_module=fake_torch)
        self.assertTrue(helper.HAS_TORCH)
        self.assertIs(helper.require_torch("Tensor processing"), fake_torch)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            helper.warn_missing_torch()
        self.assertEqual(caught, [])

    def test_torch_internal_missing_module_is_not_classified_as_absent_torch(self):
        for name in ("torch._C", "torch._prims", "numpy", "typing_extensions"):
            with self.subTest(name=name):
                error = ModuleNotFoundError(
                    "Broken Torch dependency: " + name, name=name
                )
                with self.assertRaises(ModuleNotFoundError) as caught:
                    load_optional(torch_error=error)
                self.assertIs(caught.exception, error)

    def test_broken_native_torch_or_other_import_errors_are_preserved(self):
        for error in (
            OSError("dlopen: incompatible architecture"),
            ImportError("cannot import name from torch"),
            RuntimeError("Torch initialization failed"),
        ):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)) as caught:
                    load_optional(torch_error=error)
                self.assertIs(caught.exception, error)

    def test_context_allows_explicitly_optional_absent_packages(self):
        helper = load_optional()
        for name in (
            "torch",
            "torch_geometric",
            "torch_scatter",
            "torch_sparse",
            "sklearn",
        ):
            with self.subTest(name=name):
                with helper.optional_torch_modules():
                    raise ModuleNotFoundError(
                        "No module named '" + name + "'", name=name
                    )

    def test_context_preserves_missing_internal_or_unrelated_dependencies(self):
        helper = load_optional()
        for name in (
            "torch._C",
            "torch_geometric.typo",
            "numpy",
            "easygraph_typo",
            None,
        ):
            with self.subTest(name=name):
                error = ModuleNotFoundError("Unrelated dependency failure", name=name)
                with self.assertRaises(ModuleNotFoundError) as caught:
                    with helper.optional_torch_modules():
                        raise error
                self.assertIs(caught.exception, error)

    def test_context_does_not_swallow_code_errors_or_native_library_failures(self):
        helper = load_optional()
        for error in (
            ValueError("bad configuration"),
            AttributeError("unexpected API"),
            OSError("shared library could not be loaded"),
            ImportError("incompatible Torch API"),
        ):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)) as caught:
                    with helper.optional_torch_modules():
                        raise error
                self.assertIs(caught.exception, error)

    def test_context_runs_successful_body_normally(self):
        helper = load_optional()
        visited = []
        with helper.optional_torch_modules():
            visited.append("body")
        visited.append("after")
        self.assertEqual(visited, ["body", "after"])


if __name__ == "__main__":
    unittest.main()
