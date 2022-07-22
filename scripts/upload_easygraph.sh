#!/bin/bash
# GNU/Linux
if [ ["$(expr substr $(uname -s) 1 5)" = "Linux"] ]; then
    # replace xxxx with your docker name
    docker cp xxxx:/src/Easy-Graph/dist $HOME/dist
    python -m twine upload $HOME/dist/*

# Mac OS X
else
    export WORKDIR=$HOME/build
    if [ ! -d $WORKDIR ]; then
        echo "dist is not existing"
        exit 0
    else
        cd $WORKDIR/Easy-Graph
        python -m twine upload dist/*
    fi
fi
