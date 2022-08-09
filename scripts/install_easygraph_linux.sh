#!/usr/bin/env bash
# Usage: sh install_boost_python_linux.sh -v ${python_version} -p ${python_bin} -b ${boost_version} -d ${boost_download_dir} -s ${skip_process}

# check executability
if [ ! -x "${0}" ]; then
   echo "Error: $0 is not executable."
   echo "Please run \`sudo chmod 755 $0\` and try again."
   exit 255
fi

# directories
script_dir=$(dirname "$0")
easygraph_root_dir=${script_dir}/..

# options
python_version="3.8"      # Python version. For example and by default: "3.8".
python_bin="python3"      # Python bin name or path. For example and by default: "python3".
boost_version="1.79.0"    # Boost version. For example and by default: "1.79.0".
boost_download_dir=$(pwd) # Boost download directory. By default: current working directory.
skip_process="none"       # Skip process. By default "none". Use "boost" to skip installing boost python. Use "build" to additionally skip building extension.

while getopts :v:p:b:d:s: opt; do
   case "${opt}" in
   v)
      python_version=${OPTARG}
      ;;
   p)
      python_bin=${OPTARG}
      ;;
   b)
      boost_version=${OPTARG}
      ;;
   d)
      boost_download_dir=${OPTARG%/}
      sudo mkdir -p "${boost_download_dir}"
      ;;
   s)
      skip_process=${OPTARG}
      ;;
   *)
      echo "Wrong option!"
      echo "Usage:"
      echo "  sh install_easygraph_linux.sh [options]"
      echo "Options:"
      echo "  -v        Python version. For example and by default: \"3.8\"."
      echo "  -p        Python bin name or path. For example and by default: \"python3\"."
      echo "  -b        Boost version. For example and by default: \"1.79.0\"."
      echo "  -d        Boost download directory. By default: current working directory."
      echo "  -s        Skip process. By default \"none\"."
      echo "            Use \"boost\" to skip installing boost python."
      echo "            Use \"build\" to skip building extension."
      exit 255
      ;;
   esac
done

# download, build and install boost-python
if [ "${skip_process}" != "boost" ] && [ "${skip_process}" != "build" ]; then
   install_boost_script="install_boost_python_linux.sh"
   bash "${script_dir}"/"${install_boost_script}" -v "${python_version}" -p "${python_bin}" -b "${boost_version}" -d "${boost_download_dir}"
   exit_code=${?}
   if [ ${exit_code} -ne 0 ]; then
      exit ${exit_code}
   fi
fi

# build and install easygraph
cd "${easygraph_root_dir}" || exit 255
if [ "${skip_process}" != "build" ]; then
   rm -rf build
   "${python_bin}" setup.py build_ext -l boost_python -L "/usr/local/lib" -I "/usr/local/include"
fi
sudo "${python_bin}" setup.py install
