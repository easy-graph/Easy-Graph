import setuptools
import io
import platform

with open("README.rst", "r") as fh:
    long_description = fh.read()

sources = [
    'easygraph/classes/GraphC/Graph.cpp',
    'easygraph/classes/GraphC/GraphEdge.cpp',
    'easygraph/classes/GraphC/GraphEdges.cpp',
    'easygraph/classes/GraphC/GraphEdgesIter.cpp',
    'easygraph/classes/GraphC/GraphMap.cpp',
    'easygraph/classes/GraphC/GraphMapIter.cpp',
    'easygraph/classes/GraphC/GraphModule.cpp',
    'easygraph/classes/GraphC/ModuleMethods.cpp',
]

uname = platform.uname()
compileArgs = []
if uname[0] == "Darwin" or uname[0] == "Linux":
    compileArgs = ["-std=c++11"]

setuptools.setup(name="Python-EasyGraph",
                 version="0.2a37",
                 author="Fudan MSN Group",
                 author_email="easygraph@163.com",
                 description="Easy Graph",
                 long_description=long_description,
                 long_description_content_type="text/x-rst",
                 url="https://github.com/easy-graph/Easy-Graph",
                 packages=setuptools.find_packages(),
                 classifiers=[
                     "Programming Language :: Python :: 3.6",
                     "Programming Language :: Python :: 3.7",
                     "Programming Language :: Python :: 3.8",
                     "Programming Language :: Python :: 3.9",
                     "License :: OSI Approved :: BSD License",
                     "Operating System :: OS Independent",
                 ],
                 python_requires=">=3.6, <3.10",
                 install_requires=[
                     'numpy>=1.18.5', 'tqdm>=4.49.0', 'tensorflow>=2.0.0',
                     'joblib>=0.16.0', 'six>=1.15.0', 'gensim>=4.1.2',
                     'progressbar33>=2.4', 'scikit-learn>=0.24.0',
                     'scipy>=1.5.2', 'matplotlib>=3.3.0',
                     'statsmodels>=0.12.0', 'progressbar>=2.5'
                 ],
                 ext_modules=[
                     setuptools.Extension('cpp_easygraph',
                                          sources,
                                          optional=True,
                                          extra_compile_args=compileArgs)
                 ])
