@echo off 
rem Usage: cmd install_boost_python_windows.bat -v %python_version% -p %python_bin% -b %boost_version% -d %boost_download_dir% -i %boost_install_dir% -a %compile_arch% -c "compiler_version"

rem directories
set "script_dir=%~dp0"
set "easygraph_root_dir=%script_dir%.."

rem options
set "python_version=3.8"         & rem Python version. For example and by default: "3.8".
set "python_bin=python"          & rem Python bin name or path. For example and by default: "python".
set "boost_version=1.79.0"       & rem Boost version. For example and by default: "1.79.0".
set "boost_download_dir=%cd%"    & rem Boost download directory. By default: current working directory.
set "boost_install_dir=D:\Boost" & rem Boost install directory. By default: "D:\Boost".
set "compile_arch=x86"           & rem MSVC compile arch mode referring to vcvarsall.bat arch. By default: "x86".
set "compiler_version="          & rem MSVC compiler version. By default: search for the best.

:parse
if not "%1"=="" (
    shift
    if "%~2"=="" (
        goto :wrong
    )
    if "%1"=="-v" (
        set "python_version=%~2"
        shift
        goto :parse
    )
    if "%1"=="-p" (
        set "python_bin=%~2"
        shift
        goto :parse
    )
    if "%1"=="-b" (
        set "boost_version=%~2"
        shift
        goto :parse
    )
    if "%1"=="-d" (
        set "boost_download_dir=%~2"
        shift
        goto :parse
    )
    if "%1"=="-a" (
        set "compile_arch=%~2"
        shift
        goto :parse
    )
    if "%1"=="-c" (
        set "compiler_version=%~2"
        shift
        goto :parse
    )
    if "%1"=="-i" (
        set "boost_install_dir=%~2"
        shift
        goto :parse
    )
    :wrong
        echo Wrong option!
        echo Usage:
        echo   install_boost_python_windows.bat [options]
        echo Options:
        echo   -v        Python version. For example and by default: "3.8".
        echo   -p        Python bin name or path. For example and by default: "python".
        echo   -b        Boost version. For example and by default: "1.79.0".
        echo   -d        Boost download directory. By default: current working directory.
        echo   -i        Boost install directory. By default: "D:\Boost".
        echo   -a        MSVC compile arch mode referring to vcvarsall.bat arch. By default: "x86".
        echo   -c        MSVC compiler version. By default: search for the best.
        goto :eof
    goto :parse
)

set "boost_download_dir=%boost_download_dir:/=\%"      & rem replace "/" with "\"
set "boost_install_dir=%boost_install_dir:/=\%"

rem process for bosth relative and absolute paths
set "original_dir=%cd%"
mkdir "%boost_download_dir%"
cd /d "%boost_download_dir%"
set "boost_download_dir=%cd%"
cd /d "%original_dir%"
mkdir "%boost_install_dir%"
cd /d "%boost_install_dir%"
set "boost_install_dir=%cd%"
cd /d "%original_dir%"

set "python_version_abbr=%python_version:.=%"       & rem replace "." with "", 3.8 -> 38
set "boost_version_alias=boost_%boost_version:.=_%" & rem replace "." with "_", "1.79.0" -> 1_79_0
set "boost_src_url=https://boostorg.jfrog.io/artifactory/main/release/%boost_version%/source/%boost_version_alias%.tar.gz"

rem download, build and install boost-python
cd /d %boost_download_dir%
echo Note: try to delete %boost_download_dir%\%boost_version_alias%
rmdir /s /q %boost_download_dir%\%boost_version_alias%
if exist "%boost_version_alias%.tar.gz" (
    echo Note: use existing %boost_version_alias%.tar.gz
) else (
    bitsadmin /transfer DownloadBoost%boost_version% /download /priority foreground %boost_src_url% %cd%\%boost_version_alias%.tar.gz & rem download boost source file
)
echo Note: extract %boost_version_alias%.tar.gz ...
call tar -xf %boost_version_alias%.tar.gz           & rem unzip source file

set "vcvarsallcmd="
set "installcmd="
cd /d %script_dir%
call "%python_bin%" windows_utils.py --vcvarsallcmd --arch="%compile_arch%" --version="%compiler_version%" > eg.output
set /p vcvarsallcmd=<eg.output
call "%python_bin%" windows_utils.py --installcmd --prefix="%boost_install_dir%" --version="%compiler_version%" > eg.output
set /p installcmd=<eg.output
del /q eg.output
call "%vcvarsallcmd%"                                                                     & rem init VS environment

cd /d "%boost_download_dir%"
cd /d "%boost_version_alias%"
call bootstrap.bat --with-python="%python_bin%"
echo Note: try to delete %boost_install_dir%\include
rmdir /s /q "%boost_install_dir%\include"
echo Note: try to delete %boost_install_dir%\lib
rmdir /s /q "%boost_install_dir%\lib"
echo Note: installing Boost Python in %boost_install_dir%
call %installcmd% > nul

cd /d "%original_dir%"