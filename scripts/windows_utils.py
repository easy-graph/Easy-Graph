import struct
import subprocess
import warnings


warnings.filterwarnings("ignore")  # not to disturb output
from argparse import ArgumentParser
from distutils._msvccompiler import PLAT_TO_VCVARS
from distutils._msvccompiler import _find_vcvarsall
from distutils.msvc9compiler import get_build_version
from distutils.util import get_platform


def get_vcvarsall():
    vcvarsall, _ = _find_vcvarsall(PLAT_TO_VCVARS[get_platform()])
    if vcvarsall is None:
        raise "vcvarsall.bat not found!"
    return vcvarsall


def find_best_version(version):  # priority: parameter > python built version > others
    def version_ok(version):
        out = subprocess.check_output(
            f'"{vcvarsall}" x86 -vcvars_ver={version}',
        ).decode("utf-8", errors="replace")
        return "ERROR" not in out

    if version == None or version == "":
        version = str(get_build_version())
    vcvarsall = get_vcvarsall()

    if version_ok(version):
        return version
    else:
        for i in [3, 2, 1, 0]:
            version_ = f"14.{i}"
            if version_ok(version_):
                return version_
        raise "No available msvc!"


def get_vcvarsall_cmd(arch, version):
    vcvarsall = get_vcvarsall()
    version = find_best_version(version)
    return f'"{vcvarsall}" {arch} -vcvars_ver={version}'


def get_install_cmd(prefix, version):
    version = find_best_version(version)
    toolset = "msvc-" + version
    address_model = struct.calcsize("P") * 8
    return f"""b2 install --toolset={toolset} --with-python --prefix="{prefix}" link=static runtime-link=shared address-model={address_model}"""


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--vcvarsallcmd",
        action="store_true",
        default=False,
        help="generate vcvarsall cmd",
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
    parser.add_argument(
        "--version", type=str, default=None, help="msvc compiler version"
    )
    parser.add_argument("--arch", type=str, default="x86", help="compile arch mode")

    args = parser.parse_args()
    if args.vcvarsallcmd:
        print(get_vcvarsall_cmd(args.arch, args.version))
    if args.installcmd:
        print(get_install_cmd(args.prefix, args.version))
