import os
import struct
import warnings


warnings.filterwarnings("ignore")  # not to disturb output
from argparse import ArgumentParser
from distutils._msvccompiler import MSVCCompiler
from distutils._msvccompiler import _find_exe
from distutils.msvc9compiler import get_build_version


def get_vsdevcmd_path():
    compiler = MSVCCompiler()
    compiler.initialize()
    paths = compiler._paths.split(os.pathsep)
    return _find_exe("vsdevcmd.bat", paths)


def get_install_cmd(prefix):
    toolset = "msvc-" + str(get_build_version())
    address_model = struct.calcsize("P") * 8
    return f"""b2 install --toolset={toolset} --with-python --prefix="{prefix}" link=static runtime-link=shared address-model={address_model}"""


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--vsdevcmd", action="store_true", default=False, help="search for vsdevcmd.bat"
    )
    parser.add_argument(
        "--installcmd",
        action="store_true",
        default=False,
        help="generate install command",
    )
    parser.add_argument(
        "--prefix", type=str, default=r"D:\Boost", help="boost python installation path"
    )

    args = parser.parse_args()
    if args.vsdevcmd:
        print(get_vsdevcmd_path())
    if args.installcmd:
        print(get_install_cmd(args.prefix))
