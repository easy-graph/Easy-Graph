#!/usr/bin/env python3
"""Refresh EasyGraph's static PyTorch policy from published PyPI wheels.

Run explicitly when preparing a release; package installation and imports do
not query the network. The policy records an available version interval so pip
can fall back according to wheel tags, plus the minimum macOS wheel tag. This
does not validate every libc, OS release, GPU driver, or transitive dependency.
Verify actual pip resolution and core imports on the intended operating system.
Python minor versions are checked at their .0 baseline. Only ordinary GIL
CPython wheels and compatible CPython abi3 wheels count, never cpXYt wheels.
Requires the ``packaging`` library (``python -m pip install packaging``).
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

from datetime import datetime
from datetime import timezone
from pathlib import Path


try:
    from packaging.specifiers import InvalidSpecifier
    from packaging.specifiers import SpecifierSet
    from packaging.utils import InvalidWheelFilename
    from packaging.utils import parse_wheel_filename
    from packaging.version import InvalidVersion
    from packaging.version import Version
except ImportError:
    raise SystemExit("Install the checker dependency: python -m pip install packaging")


SOURCE_URL = "https://pypi.org/pypi/torch/json"
INSTALL_URL = "https://pytorch.org/get-started/locally/"
MINIMUM_TORCH = Version("2.0")
DEFAULT_PYTHONS = ["3.{}".format(minor) for minor in range(8, 16)]
POLICY_PATH = (
    Path(__file__).resolve().parents[1] / "easygraph" / "_torch_compatibility.json"
)
TARGETS = [
    {
        "id": "linux_x86_64",
        "sys_platform": "linux",
        "platform_system": "Linux",
        "platform_machines": ["x86_64", "AMD64", "amd64"],
        "wheel_platform_suffixes": ["_x86_64"],
    },
    {
        "id": "linux_aarch64",
        "sys_platform": "linux",
        "platform_system": "Linux",
        "platform_machines": ["aarch64", "arm64", "ARM64"],
        "wheel_platform_suffixes": ["_aarch64"],
    },
    {
        "id": "windows_amd64",
        "sys_platform": "win32",
        "platform_system": "Windows",
        "platform_machines": ["AMD64", "amd64", "x86_64"],
        "wheel_platform_suffixes": ["win_amd64"],
    },
    {
        "id": "windows_arm64",
        "sys_platform": "win32",
        "platform_system": "Windows",
        "platform_machines": ["ARM64", "arm64", "aarch64"],
        "wheel_platform_suffixes": ["win_arm64"],
    },
    {
        "id": "macos_x86_64",
        "sys_platform": "darwin",
        "platform_system": "Darwin",
        "platform_machines": ["x86_64", "AMD64", "amd64"],
        "wheel_platform_suffixes": ["_x86_64", "_universal2"],
    },
    {
        "id": "macos_arm64",
        "sys_platform": "darwin",
        "platform_system": "Darwin",
        "platform_machines": ["arm64", "aarch64", "ARM64"],
        "wheel_platform_suffixes": ["_arm64", "_universal2"],
    },
]


def _python_pair(value):
    match = re.fullmatch(r"(\d+)\.(\d+)", value)
    if not match:
        raise ValueError("Python versions must be major.minor, e.g. 3.15: " + value)
    return tuple(int(part) for part in match.groups())


def _matches_platform(wheel_platform, target):
    system = target["sys_platform"]
    if system == "linux":
        valid_family = wheel_platform.startswith(("linux_", "manylinux", "musllinux"))
    elif system == "darwin":
        valid_family = wheel_platform.startswith("macosx_")
    else:
        valid_family = wheel_platform.startswith("win_")
    return valid_family and any(
        wheel_platform.endswith(suffix) for suffix in target["wheel_platform_suffixes"]
    )


def _matches_python(tag, python_pair):
    match = re.fullmatch(r"cp(\d)(\d+)", tag.interpreter)
    if not match:
        return False
    wheel_python = tuple(int(part) for part in match.groups())
    if tag.abi == "abi3":
        return wheel_python[0] == python_pair[0] and wheel_python <= python_pair
    # Older macOS torch wheels use a normal CPython interpreter and ABI "none".
    # Both forms still require this exact interpreter; cpXYt/debug are excluded.
    return wheel_python == python_pair and tag.abi in (tag.interpreter, "none")


def build_policy(metadata, python_versions, checked_at=None):
    """Return the eligible torch interval and OS floor per Python/target pair.

    ``metadata`` has the PyPI JSON API's ``releases`` mapping. Invalid release,
    filename, and Requires-Python metadata are ignored conservatively. The
    interval can contain releases with no matching wheel: pip must select the
    actual compatible wheel. minimum_macos_version is the lowest tag among all
    eligible wheels for this Python/target, not the latest torch release's floor.
    """
    python_versions = sorted(set(python_versions), key=_python_pair)
    python_pairs = {value: _python_pair(value) for value in python_versions}
    candidates = []
    for release, files in metadata.get("releases", {}).items():
        try:
            version = Version(release)
        except InvalidVersion:
            continue
        if version < MINIMUM_TORCH or version.is_prerelease or version.is_devrelease:
            continue
        for file_info in files:
            if file_info.get("packagetype") != "bdist_wheel" or file_info.get("yanked"):
                continue
            try:
                name, wheel_version, _, tags = parse_wheel_filename(
                    file_info["filename"]
                )
                requires_python = SpecifierSet(file_info.get("requires_python") or "")
            except (InvalidWheelFilename, InvalidSpecifier, KeyError, TypeError):
                continue
            if name != "torch" or wheel_version != version:
                continue
            candidates.append((version, tags, requires_python))
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)

    support = []
    for python_version in python_versions:
        for target in TARGETS:
            available_versions = set()
            macos_versions = set()
            for version, tags, requires_python in candidates:
                if not requires_python.contains(
                    python_version + ".0", prereleases=False
                ):
                    continue
                for tag in tags:
                    if not (
                        _matches_python(tag, python_pairs[python_version])
                        and _matches_platform(tag.platform, target)
                    ):
                        continue
                    if target["sys_platform"] == "darwin":
                        match = re.match(r"macosx_(\d+)_(\d+)_", tag.platform)
                        if not match:
                            continue
                        macos_versions.add(tuple(int(part) for part in match.groups()))
                    available_versions.add(version)
            support.append(
                {
                    "python_version": python_version,
                    "target_id": target["id"],
                    "torch_version": str(max(available_versions))
                    if available_versions
                    else None,
                    "minimum_torch_version": str(min(available_versions))
                    if available_versions
                    else None,
                    "minimum_macos_version": (
                        "{}.{}".format(*min(macos_versions)) if macos_versions else None
                    ),
                }
            )
    return {
        "schema_version": 1,
        "checked_at": checked_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_url": SOURCE_URL,
        "official_install_url": INSTALL_URL,
        "minimum_torch_version": str(MINIMUM_TORCH),
        "availability_scope": (
            "PyPI wheel platform/CPU family, Python .0 baseline, available torch"
            " interval, and minimum macOS wheel tag. Pip must select a compatible"
            " wheel; actual resolution, core imports, libc, GPU drivers, and transitive"
            " dependencies require validation on the intended operating system."
        ),
        "python_versions": python_versions,
        "targets": TARGETS,
        "support": support,
    }


def fetch_metadata():
    request = urllib.request.Request(
        SOURCE_URL,
        headers={
            "User-Agent": "EasyGraph-PyTorch-compatibility-checker/1",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def write_policy(policy, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")


def print_table(policy):
    print(
        "Python  Target             PyPI torch interval   macOS floor  Default policy"
    )
    print(
        "------  -----------------  --------------------  -----------  --------------"
    )
    for row in policy["support"]:
        version = row["torch_version"]
        interval = (
            ">={},<={}".format(row["minimum_torch_version"], version)
            if version
            else "no wheel"
        )
        print(
            "{:<6}  {:<17}  {:<20}  {:<11}  {}".format(
                row["python_version"],
                row["target_id"],
                interval,
                row["minimum_macos_version"] or "-",
                "eligible; pip selects wheel"
                if version
                else "no; see official installer",
            )
        )
    print("\nNo PyPI wheel does not mean PyTorch is unsupported.")
    print("Check official installation options: " + INSTALL_URL)
    print(
        "Python minor versions are checked at their .0 baseline. An available version"
    )
    print(
        "interval permits fallback; pip resolution and core imports still need"
        " validation"
    )
    print("on the intended OS, including libc and transitive dependency requirements.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python-versions", nargs="+", default=DEFAULT_PYTHONS, metavar="MAJOR.MINOR"
    )
    parser.add_argument(
        "--output", type=Path, help="write the resulting JSON to this path"
    )
    parser.add_argument(
        "--write-policy",
        action="store_true",
        help="explicitly refresh easygraph/_torch_compatibility.json",
    )
    parser.add_argument(
        "--pypi-json",
        type=Path,
        help="read a previously saved PyPI JSON response instead of fetching",
    )
    args = parser.parse_args(argv)
    try:
        metadata = (
            json.loads(args.pypi_json.read_text(encoding="utf-8"))
            if args.pypi_json
            else fetch_metadata()
        )
        policy = build_policy(metadata, args.python_versions)
        if args.output:
            write_policy(policy, args.output)
        if args.write_policy:
            write_policy(policy, POLICY_PATH)
    except (OSError, ValueError, urllib.error.URLError) as error:
        parser.exit(1, "PyTorch compatibility check failed: {}\n".format(error))
    print_table(policy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
