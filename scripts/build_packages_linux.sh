#!/bin/bash
cd /src/Easy-Graph/ || exit

# # compile Python 3.6
# PYTHON3=/opt/python/cp36-cp36m/bin/python3
# time $PYTHON3 setup.py bdist_wheel
# mv dist/*-linux_* "$(ls dist/*-linux_* | sed -e "s/-linux_/-manylinux1_/g")"

# compile Python 3.7
PYTHON3=/opt/python/cp37-cp37m/bin/python3
time $PYTHON3 setup.py bdist_wheel
mv dist/*-linux_* "$(ls dist/*-linux_* | sed -e "s/-linux_/-manylinux1_/g")"

# compile Python 3.8
PYTHON3=/opt/python/cp38-cp38/bin/python3
time $PYTHON3 setup.py bdist_wheel
mv dist/*-linux_* "$(ls dist/*-linux_* | sed -e "s/-linux_/-manylinux1_/g")"

# compile Python 3.9
PYTHON3=/opt/python/cp39-cp39/bin/python3
time $PYTHON3 setup.py bdist_wheel
mv dist/*-linux_* "$(ls dist/*-linux_* | sed -e "s/-linux_/-manylinux1_/g")"
