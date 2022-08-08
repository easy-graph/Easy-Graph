#!/bin/bash -e
# Usage: sh install_boost_python_linux.sh -v ${python_version} -p ${python_bin} -b ${boost_version} -d ${boost_download_dir}
if [ ! -x ${0} ]
then
   echo "Error: $0 is not executable."
   echo "Please run \`sudo chmod 755 $0\` and try again."
   exit 255
fi

# check executability
current_dir=$(pwd)
script_dir=$(dirname "$0")
easygraph_root_dir=${script_dir}/..

# directories
python_version="3.8" # Python version. For example and by default: \"3.8\".
python_bin="python3" # Python bin name or path. For example and by default: \"python3\".
boost_version="1.79.0" # Boost version. For example and by default: \"1.79.0\".
boost_download_dir=$(pwd) # Boost download directory. By default: current working directory.

while getopts :v:p:b:d: opt
do 
    case "${opt}" in 
        v) python_version=${OPTARG}
           ;; 
        p) python_bin=${OPTARG}
           ;; 
        b) boost_version=${OPTARG}
           ;;
        d) boost_download_dir=${OPTARG%/}
           sudo mkdir -p $boost_download_dir
           ;;
        *) echo "Wrong option!"
           echo "Usage:"
           echo "  chmod 755 install_boost_python_linux.sh"
           echo "  sh install_boost_python_linux.sh [options]"
           echo "Options:"
           echo "  -v        Python version. For example and by default: \"3.8\"."
           echo "  -p        Python bin name or path. For example and by default: \"python3\"."
           echo "  -b        Boost version. For example and by default: \"1.79.0\"."
           echo "  -d        Boost download directory. By default: current working directory."
           exit 255
           ;;
    esac 
done

python_version_abbr=${python_version//./} # replace "." with "", 3.8 -> 38
boost_version_alias=boost_${boost_version//./_} # replace "." with "_", "1.79.0" -> 1_79_0
boost_src_url="https://boostorg.jfrog.io/artifactory/main/release/${boost_version}/source/${boost_version_alias}.tar.gz"

# install python3-dev, gcc, g++
sudo apt-get install python3-dev # high version dev file can be found on: e.g. https://pkgs.org/download/python3.10-dev
sudo apt-get install gcc -y
sudo apt-get install g++ -y

# download, build and install boost-python
cd ${boost_download_dir}
rm -rf ${boost_version_alias}
if [ -f "${boost_version_alias}.tar.gz" ]
then 
   echo "Note: use existing ${boost_version_alias}.tar.gz ."
else
   wget ${boost_src_url} # download boost source file
fi
tar -xf ${boost_version_alias}.tar.gz # unzip cource file
sudo rm -rf /usr/local/lib/libboost* # delete existing boost python libs
sudo rm -rf /usr/local/include/boost # delete existing boost python header files
cd ${boost_version_alias}
./bootstrap.sh --with-python=${python_bin} # compile boost compiler b2
sudo ./b2 cxxflags="-fPIC" install --with-python  # use b2 to compile and install boost python
sudo ln -s /usr/local/lib/libboost_python${python_version_abbr}.a /usr/local/lib/libboost_python.a # soft link libboost_pythonxx.a to libboost_python.a for using static library but dynamic one in later linking