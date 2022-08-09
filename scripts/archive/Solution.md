# Solution Doc

This doc aims to record the problem encountered during packaging.

## pip

### Windows


### Mac

* Q: ERROR: Package 'python-easygraph' requires a different Python: 3.9.10 not in '<=3.9,>=3.6'

  A : you might need to update dependencies to compatibility: https://stackoverflow.com/questions/66593103/why-does-pip-claim-that-a-version-of-python-is-not-in-a-given-range

### Linux

## conda

### Windows

* Q: WARNING:conda.gateways.disk.delete:Could not remove or rename ....\cudnn-8.2.1.32-h754d62a_0.tar.bz2.  Please remove this file manually (you may need to reboot to free file handles)

  A: https://stackoverflow.com/questions/62215850/warning-conda-gateways-diskunlink-or-rename-to-trash140

### Mac


### Linux
