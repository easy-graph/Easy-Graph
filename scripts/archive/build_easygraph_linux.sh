#!/bin/bash
# Install docker:
#       https://yeasy.gitbook.io/docker_practice/install/ubuntu#shi-yong-jiao-ben-zi-dong-an-zhuang

# compiles mac packages for python 3.6、3.7、3.8、3.9
# requirements:
#       docker
#       quay.io/pypa/manylinux1_x86_64 (docker container)

export WORKDIR=$HOME/build
if [ ! -d "$WORKDIR" ]; then
    mkdir -p "$WORKDIR"
    cd "$WORKDIR" || exit
    git clone https://github.com/easy-graph/Easy-Graph.git
fi

chmod -R 777 "$WORKDIR"/Easy-Graph/
cd "$WORKDIR"/Easy-Graph || exit

docker run -it -v "$WORKDIR":/src --user "$(id -u):$(id -g)" quay.io/pypa/manylinux1_x86_64 /src/Easy-Graph/scripts/build_packages_linux.sh
