REM @echo off
REM requirements:
REM     conda
REM
REM create envs:
    REM conda create -n py36 python=3.6.8
    REM conda create -n py37 python=3.7.9
    REM conda create -n py38 python=3.8.5
    REM conda create -n py39 python=3.9.6

cd ../

REM compile Python 3.6.8
call conda activate py36
call python setup.py bdist_wheel

REM compile Python 3.7.9
call conda activate py37
call python setup.py bdist_wheel

REM compile Python 3.8.5
call conda activate py38
call python setup.py bdist_wheel

REM compile Python 3.9.6
call conda activate py39
call python setup.py bdist_wheel

pause
