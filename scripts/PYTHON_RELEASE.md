# Preparing EasyGraph for a new Python version

EasyGraph installs a compatible PyTorch release automatically when its saved policy
lists an ordinary CPython wheel for the user's Python minor, OS and CPU family
on PyPI. Where no such wheel is listed, the default installation omits PyTorch.
Core graph algorithms and the C++ extension remain usable. Importing EasyGraph
without PyTorch emits one warning with the official installation page:
https://pytorch.org/get-started/locally/.

The policy is a release-time snapshot, not a network check during installation
or import. Updating the file does not change already-published wheels: publish a
new EasyGraph version after checking the resulting installation.

## Refresh the policy

From the repository root, run:

```sh
python -m pip install packaging
python scripts/check_pytorch_support.py \
  --python-versions 3.8 3.9 3.10 3.11 3.12 3.13 3.14 3.15 \
  --write-policy
```

Append the new Python minor to this command when it is released. The checker
uses https://pypi.org/pypi/torch/json and only counts stable, non-yanked torch>=2.0
wheels matching the Python version, architecture and Requires-Python metadata.
It handles both ordinary CPython ABI tags and older macOS `cpXY-none` wheels.
Free-threaded/debug wheels do not enable ordinary CPython dependencies.

No PyPI wheel does **not** mean PyTorch has no compatible build: its official
index can provide additional builds. For example, PyTorch's Python 3.15 builds
are currently distributed through the official index rather than PyPI. After
installing one, restart Python; EasyGraph will detect and use it normally.

The table checks platform/CPU families at the Python minor's `.0` baseline.
It does not prove support for every OS release, libc, GPU or dependent package.
Default Linux PyTorch dependencies are omitted when building on musl/Alpine;
the current PyPI PyTorch Linux wheels require glibc. Keep the same libc family
when building and installing the corresponding platform wheel.

## Update and validate the release

1. Review `easygraph/_torch_compatibility.json`; every non-null row becomes a
   platform- and Python-specific PyTorch version interval in `setup.py`. Pip can
   choose a compatible older wheel based on OS/libc tags. macOS default PyTorch
   markers also require the minimum OS family observed in the available wheels.
   Intervals that include PyTorch before 2.3 require NumPy<2 for compatibility
   with its older binary ABI; newer Python intervals retain NumPy 2 support.
2. Update EasyGraph's version, `python_requires` upper bound, classifier and
   `cpXY-*` selector in `pyproject.toml`. Retain the existing minimum Python
   version unless dropping it is an explicit project decision.
3. Select a cibuildwheel release that supports the new interpreter. Keep a
   legacy builder if the latest version has dropped older supported versions.
4. Build the wheel and source distribution. Install the wheel into a fresh
   environment **without** manually installing torch. Verify that the expected
   dependency is installed or omitted automatically.
5. Run `python scripts/wheel_smoke.py` outside the source directory. It checks
   one missing-Torch warning, Graph/DiGraph and C++ conversion, degree centrality,
   Python and C++ PageRank, SciPy conversion, and either PyTorch hypergraph
   message passing or a clear dependency error with the installation URL.
6. Run the same checks on Linux, Windows, macOS Intel and macOS Apple Silicon.
   Then manually dispatch `wheel_publish.yml` to publish. That workflow uploads
   to PyPI after all builds and smoke checks pass.

The `[torch]` extra remains an explicit request for PyTorch and graph-learning
dependencies. Installing that extra on a platform with no PyPI PyTorch wheel
can fail during dependency resolution; use the official installer first where
necessary. Optional `torch_geometric`, `torch_scatter` and scikit-learn features
need their own dependencies; the core smoke check does not claim full ML coverage.
