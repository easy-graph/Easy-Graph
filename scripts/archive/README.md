# build and upload easygraph
This doc aims to help you with building and uploading easygraph in different OS using pip and conda.

## Using Pip

### Windows

#### build easygraph

1.Get the source code and cd to your project

    git clone https://github.com/easy-graph/Easy-Graph.git
    cd to your project

2.run the script in powershell

    ./build_easygraph_windows.bat
    【Note】 The premise is that you have to install prerequisites according to the comments in the script.



#### upload easygraph

    cd to your project
    python -m twine upload dist/*

### MacOS

#### build easygraph

1.Get the source code and cd to your project

    git clone https://github.com/easy-graph/Easy-Graph.git
    cd to your project

2.run the script

    bash/zsh scripts/build_easygraph_macos.sh
    【Note】 The premise is that you have to install prerequisites according to the comments in the script.



#### upload easygraph

run the script:

    bash/zsh scripts/upload_easygraph.sh

### Linux

#### build easygraph

1.Get the source code and cd to your project

    git clone https://github.com/easy-graph/Easy-Graph.git
    cd /Easy-Graph

2.run the script:

    bash/zsh scripts/build_easygraph_linux.sh
    【Note】 The premise is that you have to install prerequisites according to the comments in the script.

#### upload easygraph

modify the **docker name** in upload_easygraph.sh, then:

    bash/zsh scripts/upload_easygraph.sh



### Verify
The verify process is the same in all OS.
1. pip cache purge
2. pip uninstall Python-EasyGraph
3. pip install -i https://pypi.Python.org/simple/ Python-EasyGraph

If the download file is whl, it means that your build&upload process is successful.


## Using conda

Refer to https://conda.io/projects/conda-build/en/latest/user-guide/tutorials/build-pkgs-skeleton.html

### run the commands below in Mac, Linux and Windows
    conda build --python 3.6 Easy-Graph
    conda build --python 3.7 Easy-Graph
    conda build --python 3.8 Easy-Graph
    conda build --python 3.9 Easy-Graph

Finaly, you are free to change all the script to your taste.
