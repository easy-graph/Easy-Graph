"""Offline tests of PyPI wheel selection; no EasyGraph imports or network."""

import importlib.util
import unittest

from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "check_pytorch_support",
    Path(__file__).resolve().parents[1] / "check_pytorch_support.py",
)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def wheel(
    version,
    interpreter="cp315",
    abi="cp315",
    platform="macosx_14_0_arm64",
    requires_python=">=3.8",
    yanked=False,
):
    return {
        "filename": "torch-{}-{}-{}-{}.whl".format(version, interpreter, abi, platform),
        "packagetype": "bdist_wheel",
        "requires_python": requires_python,
        "yanked": yanked,
    }


def selected_row(metadata, python="3.15", target="macos_arm64"):
    policy = checker.build_policy(
        metadata, [python], checked_at="2026-10-12T00:00:00+00:00"
    )
    return next(row for row in policy["support"] if row["target_id"] == target)


def selected(metadata, python="3.15", target="macos_arm64"):
    return selected_row(metadata, python, target)["torch_version"]


class PyTorchSupportTests(unittest.TestCase):
    def test_latest_stable_non_yanked_release_and_minimum(self):
        metadata = {
            "releases": {
                "1.13.0": [wheel("1.13.0")],
                "2.9.0": [wheel("2.9.0")],
                "2.10.0": [wheel("2.10.0")],
                "2.11.0": [wheel("2.11.0", yanked=True)],
                "2.12.0rc1": [wheel("2.12.0rc1")],
                "2.12.0.dev1": [wheel("2.12.0.dev1")],
                "invalid": [wheel("2.13.0")],
            }
        }
        self.assertEqual(selected(metadata), "2.10.0")
        self.assertIsNone(selected({"releases": {"1.13.0": [wheel("1.13.0")]}}))

    def test_free_threaded_and_debug_wheels_do_not_count(self):
        for interpreter, abi in [
            ("cp315", "cp315t"),
            ("cp315t", "cp315t"),
            ("cp315", "cp315d"),
        ]:
            with self.subTest(interpreter=interpreter, abi=abi):
                metadata = {"releases": {"2.10.0": [wheel("2.10.0", interpreter, abi)]}}
                self.assertIsNone(selected(metadata))
        self.assertEqual(
            selected({"releases": {"2.10.0": [wheel("2.10.0")]}}), "2.10.0"
        )

    def test_abi3_supports_newer_cpython_only(self):
        metadata = {"releases": {"2.10.0": [wheel("2.10.0", "cp39", "abi3")]}}
        self.assertEqual(selected(metadata, python="3.15"), "2.10.0")
        self.assertEqual(selected(metadata, python="3.9"), "2.10.0")
        self.assertIsNone(selected(metadata, python="3.8"))
        self.assertIsNone(selected(metadata, python="4.0"))

    def test_older_macos_cpython_none_abi_wheels(self):
        # Real PyPI format: torch-2.4.1-cp38-none-macosx_11_0_arm64.whl.
        metadata = {
            "releases": {
                "2.4.1": [
                    wheel(
                        "2.4.1",
                        "cp38",
                        "none",
                        "macosx_11_0_arm64",
                        requires_python=">=3.8.0",
                    )
                ]
            }
        }
        self.assertEqual(selected(metadata, python="3.8"), "2.4.1")
        self.assertIsNone(selected(metadata, python="3.9"))
        self.assertIsNone(selected(metadata, python="3.8", target="macos_x86_64"))

    def test_version_interval_and_macos_floor_allow_older_os_fallback(self):
        metadata = {
            "releases": {
                "2.0.0": [
                    wheel("2.0.0", "cp311", "none", "macosx_11_0_arm64", yanked=True)
                ],
                "2.1.0": [
                    wheel(
                        "2.1.0",
                        "cp311",
                        "none",
                        "macosx_10_9_arm64",
                        requires_python=">=3.12",
                    )
                ],
                "2.2.2": [wheel("2.2.2", "cp311", "none", "macosx_11_0_arm64")],
                "2.14.1": [wheel("2.14.1", "cp311", "cp311", "macosx_14_0_arm64")],
                "2.15.0rc1": [
                    wheel("2.15.0rc1", "cp311", "cp311", "macosx_10_9_arm64")
                ],
            }
        }
        row = selected_row(metadata, python="3.11")
        self.assertEqual(row["torch_version"], "2.14.1")
        self.assertEqual(row["minimum_torch_version"], "2.2.2")
        self.assertEqual(row["minimum_macos_version"], "11.0")

    def test_macos_minimum_uses_numeric_tag_order_and_universal2(self):
        metadata = {
            "releases": {
                "2.2.2": [wheel("2.2.2", platform="macosx_10_9_universal2")],
                "2.10.0": [wheel("2.10.0", platform="macosx_11_0_arm64")],
                "2.11.0": [wheel("2.11.0", platform="macosx_14_0_universal2")],
            }
        }
        for target in ("macos_x86_64", "macos_arm64"):
            with self.subTest(target=target):
                row = selected_row(metadata, target=target)
                self.assertEqual(row["minimum_torch_version"], "2.2.2")
                self.assertEqual(row["torch_version"], "2.11.0")
                self.assertEqual(row["minimum_macos_version"], "10.9")

    def test_other_platform_has_version_interval_without_macos_floor(self):
        metadata = {
            "releases": {
                "2.0.0": [wheel("2.0.0", platform="manylinux2014_x86_64")],
                "2.14.1": [wheel("2.14.1", platform="manylinux_2_28_x86_64")],
            }
        }
        row = selected_row(metadata, target="linux_x86_64")
        self.assertEqual(row["minimum_torch_version"], "2.0.0")
        self.assertEqual(row["torch_version"], "2.14.1")
        self.assertIsNone(row["minimum_macos_version"])

    def test_no_matching_wheel_sets_all_availability_fields_to_null(self):
        row = selected_row({"releases": {}})
        for field in (
            "torch_version",
            "minimum_torch_version",
            "minimum_macos_version",
        ):
            self.assertIsNone(row[field])

    def test_requires_python_applies_to_each_file_at_patch_zero(self):
        for requirement in (">=3.16", "<3.15", ">=3.15.1", "not valid"):
            with self.subTest(requirement=requirement):
                metadata = {
                    "releases": {
                        "2.10.0": [wheel("2.10.0", requires_python=requirement)]
                    }
                }
                self.assertIsNone(selected(metadata))
        for requirement in (">=3.15,<3.16", None):
            with self.subTest(requirement=requirement):
                metadata = {
                    "releases": {
                        "2.10.0": [wheel("2.10.0", requires_python=requirement)]
                    }
                }
                self.assertEqual(selected(metadata), "2.10.0")

    def test_platform_and_cpu_families(self):
        platforms = {
            "linux_x86_64": "manylinux_2_28_x86_64",
            "linux_aarch64": "manylinux_2_28_aarch64",
            "windows_amd64": "win_amd64",
            "windows_arm64": "win_arm64",
            "macos_x86_64": "macosx_10_9_x86_64",
            "macos_arm64": "macosx_14_0_arm64",
        }
        for wanted, platform in platforms.items():
            metadata = {"releases": {"2.10.0": [wheel("2.10.0", platform=platform)]}}
            for target in platforms:
                with self.subTest(wanted=wanted, target=target):
                    self.assertEqual(
                        selected(metadata, target=target),
                        "2.10.0" if target == wanted else None,
                    )

    def test_universal2_counts_for_both_mac_architectures(self):
        metadata = {
            "releases": {"2.10.0": [wheel("2.10.0", platform="macosx_11_0_universal2")]}
        }
        self.assertEqual(selected(metadata, target="macos_arm64"), "2.10.0")
        self.assertEqual(selected(metadata, target="macos_x86_64"), "2.10.0")
        self.assertIsNone(selected(metadata, target="linux_aarch64"))

    def test_compressed_platform_tags(self):
        metadata = {
            "releases": {
                "2.10.0": [
                    wheel(
                        "2.10.0", platform="manylinux_2_17_x86_64.manylinux2014_x86_64"
                    )
                ]
            }
        }
        self.assertEqual(selected(metadata, target="linux_x86_64"), "2.10.0")

    def test_sdist_pure_python_invalid_filename_and_wrong_version_are_ignored(self):
        files = [
            {"filename": "torch-2.10.0.tar.gz", "packagetype": "sdist"},
            wheel("2.10.0", "py3", "none", "any"),
            wheel("2.10.0", "cp314", "cp314"),
            wheel("2.9.0"),
            {"filename": "broken.whl", "packagetype": "bdist_wheel"},
        ]
        self.assertIsNone(selected({"releases": {"2.10.0": files}}))

    def test_schema_and_sorted_versions(self):
        policy = checker.build_policy(
            {"releases": {}},
            ["3.15", "3.8", "3.8"],
            checked_at="2026-10-12T00:00:00+00:00",
        )
        self.assertEqual(policy["schema_version"], 1)
        self.assertEqual(policy["python_versions"], ["3.8", "3.15"])
        self.assertEqual(len(policy["targets"]), 6)
        self.assertEqual(len(policy["support"]), 12)
        self.assertEqual(policy["source_url"], checker.SOURCE_URL)
        self.assertEqual(policy["official_install_url"], checker.INSTALL_URL)
        with self.assertRaises(ValueError):
            checker.build_policy({"releases": {}}, ["3.15.0"])


if __name__ == "__main__":
    unittest.main()
