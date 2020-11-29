import setuptools
import io

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Python-EasyGraph",                                     
    version="0.2a24",                                        
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
        "License :: OSI Approved :: BSD License",           
        "Operating System :: OS Independent",               
    ],
    python_requires=">=3.6,<3.9",
    install_requires=['numpy>=1.18.5,<1.19.0','tqdm>=4.49.0','tensorflow>=2.0.0','joblib>=0.16.0','six>=1.15.0','gensim>=3.8.3','progressbar33>=2.4','scikit-learn>=0.23.2','scipy>=1.5.2',
    ]
)

