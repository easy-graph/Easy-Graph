import os
import platform
import subprocess
import sys

from distutils import sysconfig
from pathlib import Path

import setuptools

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import build_ext

disable_cpp = False
if "--disable-cpp" in sys.argv:
    disable_cpp = True
    sys.argv.remove("--disable-cpp")

enable_gpu = False
if "--enable-gpu" in sys.argv:
    enable_gpu = True
    sys.argv.remove("--enable-gpu")

class CMakeExtension(setuptools.Extension):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, sources=[], **kwargs)

class CMakeBuild(build_ext):
    def build_extension(self, ext: build_ext) -> None:
        global disable_cpp
        global enable_gpu

        if disable_cpp:
            return
        
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DEASYGRAPH_ENABLE_GPU={'ON' if enable_gpu else 'OFF'}"
        ]
        eg_dir_path = Path("./").resolve()
        subprocess.run(
            ["cmake", ".", *cmake_args], cwd=eg_dir_path, check=True
        )
        subprocess.run(
            ["cmake", "--build", "."], cwd=eg_dir_path, check=True
        )

with open("README.md") as fh:
    long_description = fh.read()

CYTHON_STR = "Cython"
setuptools.setup(
    name="Python-EasyGraph",
    version="1.3",
    author="Fudan DataNET Group",
    author_email="mgao21@m.fudan.edu.cn",
    description="Easy Graph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/easy-graph/Easy-Graph",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8, <3.13",
    install_requires=[
        "numpy>=1.23.1; python_version>='3.10'",
        "numpy>=1.19.5; python_version>='3.7' and python_version<='3.12'",
        "tqdm>=4.49.0",
        "joblib>=1.2.0",
        "six>=1.15.0, <1.16.0",
        "gensim>=4.1.2; python_version>='3.7'",
        "progressbar33>=2.4",
        "scikit-learn>=0.24.0, <=1.0.2; python_version=='3.7'",
        "scikit-learn>=0.24.0; python_version>='3.8'",
        "scipy>=1.5.0, <=1.7.3; python_version=='3.7'",
        "scipy>=1.8.0; python_version>='3.8'",
        "statsmodels>=0.12.0; python_version>='3.7'",
        "progressbar>=2.5",
        "nose>=0.10.1",
        "pandas>=1.0.1, <=1.1.5; python_version<='3.7'",
        "matplotlib",
        "requests",
        "optuna",
    ],
    setup_requires=[CYTHON_STR],
    test_suite="nose.collector",
    tests_require=[],
    cmdclass={
        "build_ext": CMakeBuild,
    },
    ext_modules=[
        CMakeExtension("cpp_easygraph", optional=True)
    ],
)
