#!/usr/bin/env bash

export PYBIND11_EXT_LIBRARIES='stdc++'
export PYBIND11_EXT_INCLUDE_DIRS='/usr/local/Cellar/gcc/12.2.0/include/c++/12:/usr/local/Cellar/gcc/12.2.0/include/c++/12/x86_64-apple-darwin21'

CC='/usr/local/Cellar/gcc/12.2.0/bin/gcc-12' CXX='/usr/local/Cellar/gcc/12.2.0/bin/g++-12' python3 setup.py build_ext
python3 setup.py install --skip-build
python3 setup.py bdist_wheel

CC='/usr/local/Cellar/gcc/12.2.0/bin/gcc-12' CXX='/usr/local/Cellar/gcc/12.2.0/bin/g++-12' python3.9 setup.py build_ext
CC='/usr/local/Cellar/gcc/12.2.0/bin/gcc-12' CXX='/usr/local/Cellar/gcc/12.2.0/bin/g++-12' python3.8 setup.py build_ext
CC='/usr/local/Cellar/gcc/12.2.0/bin/gcc-12' CXX='/usr/local/Cellar/gcc/12.2.0/bin/g++-12' python3.7 setup.py build_ext
python3.9 setup.py bdist_wheel
python3.8 setup.py bdist_wheel
python3.7 setup.py bdist_wheel
