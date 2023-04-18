import os
import platform

from distutils import sysconfig
from pathlib import Path

import setuptools

# print(setuptools.__file__)
# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension
from pybind11.setup_helpers import build_ext


with open("README.rst") as fh:
    long_description = fh.read()

cpp_source_dir = Path(__file__).parent / "cpp_easygraph"
sources = list(str(x) for x in cpp_source_dir.rglob("*.cpp"))

uname = platform.uname()


compileArgs = []
if uname[0] == "Darwin" or uname[0] == "Linux":
    compileArgs = ["-std=c++11"]
CYTHON_STR = "Cython"
setuptools.setup(
    name="Python-EasyGraph",
    version="0.2a46",
    author="Fudan MSN Group",
    author_email="easygraph@163.com",
    description="Easy Graph",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/easy-graph/Easy-Graph",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7, <3.11",
    install_requires=[
        "numpy>=1.23.1; python_version>='3.10'",
        "numpy>=1.19.5; python_version>='3.7' and python_version<'3.10'",
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
    ],
    setup_requires=[CYTHON_STR],
    test_suite="nose.collector",
    tests_require=[],
    cmdclass={"build_ext": build_ext},
    ext_modules=[
        Pybind11Extension(
            "cpp_easygraph", sources, optional=True, extra_compile_args=compileArgs,
        )
    ],
)
