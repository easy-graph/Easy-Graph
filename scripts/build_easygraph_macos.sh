#!/usr/bin/env bash

# Install pyenv：
#       brew install pyenv: https://segmentfault.com/a/1190000016819226
#
# compiles mac packages for python 3.6、3.7、3.8、3.9
# requirements:
#       pyenv

export WORKDIR=$HOME/build
if [ ! -d "$WORKDIR" ]; then
    mkdir -p "$WORKDIR"
    cd "$WORKDIR" || exit
    git clone https://github.com/easy-graph/Easy-Graph.git
fi
cd "$WORKDIR"/Easy-Graph || exit

pip install wheel

# # compile Python 3.6
# pyenv global 3.6.15
# time python setup.py bdist_wheel --plat-name macosx-10.9-x86_64

# compile Python 3.7
pyenv global 3.7.9
time python setup.py bdist_wheel --plat-name macosx-10.9-x86_64

# compile Python 3.8
pyenv global 3.8.5
time python setup.py bdist_wheel --plat-name macosx-10.9-x86_64

# compile Python 3.9
pyenv global 3.9.10
time python setup.py bdist_wheel --plat-name macosx-10.9-x86_64
