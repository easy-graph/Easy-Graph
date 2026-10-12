import json
import os
import platform
import re
import subprocess
import sys

from distutils import sysconfig
from pathlib import Path
from packaging.version import Version

import setuptools

from setuptools.command.build_ext import build_ext


# The following code is maily from https://github.com/pybind/cmake_example/tree/master

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

class CMakeExtension(setuptools.Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())

class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        enable_gpu = "EASYGRAPH_ENABLE_GPU" in os.environ and "TRUE" in os.environ.get("EASYGRAPH_ENABLE_GPU").upper()

        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        cfg = "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DPYBIND11_FINDPYTHON=ON",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            f"-DEASYGRAPH_ENABLE_GPU={'ON' if enable_gpu else 'OFF'}",
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if self.compiler.compiler_type == "msvc":
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]


        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

CYTHON_STR = "Cython"


def default_torch_requirements():
    # Refresh explicitly with scripts/check_pytorch_support.py before releases.
    # Never query the network while building package metadata.
    policy_path = Path(__file__).parent / "easygraph" / "_torch_compatibility.json"
    with policy_path.open(encoding="utf-8") as stream:
        policy = json.load(stream)
    if policy["schema_version"] != 1:
        raise ValueError("Unsupported PyTorch compatibility policy schema")
    targets = {target["id"]: target for target in policy["targets"]}
    requirements = []
    for row in policy["support"]:
        if row["torch_version"] is None:
            continue
        target = targets[row["target_id"]]
        # PyPI's Linux torch wheels target glibc, not musl/Alpine.
        if (target["sys_platform"] == "linux" and sys.platform == "linux"
                and platform.libc_ver()[0] != "glibc"):
            continue
        machines = " or ".join(
            "platform_machine == '{}'".format(machine)
            for machine in target["platform_machines"]
        )
        marker = (
            "python_version == '{}' and sys_platform == '{}' and "
            "platform_system == '{}' and platform_python_implementation == 'CPython' "
            "and ({})"
        ).format(row["python_version"], target["sys_platform"],
                 target["platform_system"], machines)
        if target["sys_platform"] == "darwin":
            major, minor = map(int, row["minimum_macos_version"].split("."))
            darwin_major = major + 9 if major >= 11 else minor + 4
            marker += " and platform_release >= '{}'".format(darwin_major)
        minimum = row["minimum_torch_version"]
        requirements.append("torch>={},<={}; {}".format(
            minimum, row["torch_version"], marker
        ))
        # Older macOS torch binaries use NumPy 1's C ABI.
        if Version(minimum) < Version("2.3"):
            requirements.append("numpy<2; {}".format(marker))
    return requirements


setuptools.setup(
    name="Python-EasyGraph",
    version="1.6.3",
    author="Fudan DataNET Group",
    author_email="mgao21@m.fudan.edu.cn",
    description="Easy Graph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/easy-graph/Easy-Graph",
    packages=setuptools.find_packages(),
    package_data={"easygraph": ["_torch_compatibility.json"]},
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: 3.15",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8, <3.16",
    install_requires=[
        "numpy>=1.23; python_version < '3.14'",
        "numpy>=2.0; python_version >= '3.14'",
        "tqdm>=4.49.0",
        "joblib>=1.2.0",
        "six>=1.16.0",
        "gensim>=4.3.3; python_version < '3.14'",
        "progressbar33>=2.4",
        "scikit-learn>=0.24.0, <=1.0.2; python_version=='3.7'",
        "scikit-learn>=0.24.0; python_version>='3.8' and python_version<'3.14'",
        "scipy>=1.5.0, <=1.7.3; python_version=='3.7'",
        "scipy>=1.8.0; python_version>='3.8' and python_version<'3.14'",
        "scipy>=1.18.1; python_version>='3.14'",
        "statsmodels>=0.12.0; python_version>='3.7' and python_version<'3.14'",
        "progressbar>=2.5",
        "nose>=0.10.1",
        "pandas>=1.0.1, <=1.1.5; python_version<='3.7'",
        "pandas>=1.1.0; python_version>='3.8' and python_version<'3.14'",
        "matplotlib",
        "requests",
        "optuna",
        "fastjsonschema",
    ] + default_torch_requirements(),
    extras_require={
        "torch": [
            "torch>=2.0",
            "fastjsonschema",
            "torch_geometric>=2.3",
        ],
    },
    setup_requires=[CYTHON_STR],
    cmdclass={
        "build_ext": CMakeBuild,
    },
    ext_modules=[
        CMakeExtension("cpp_easygraph")
    ],
)
