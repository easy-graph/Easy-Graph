#!/usr/bin/env bash

alias 'gcc'='/usr/local/Cellar/gcc/12.2.0/bin/gcc-12'
alias 'g++'='/usr/local/Cellar/gcc/12.2.0/bin/g++-12'

CC='/usr/local/Cellar/gcc/12.2.0/bin/gcc-12' CXX='/usr/local/Cellar/gcc/12.2.0/bin/g++-12' python3 setup.py build_ext
python3 setup.py install --skip-build

