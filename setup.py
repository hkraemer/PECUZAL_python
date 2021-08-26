import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pecuzal_embedding", 
    version="1.3.4",
    author="K.H.Kraemer",
    author_email="hkraemer@pik-potsdam.de",
    description="PECUZAL automatic embedding of uni- and multivariate time series",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/hkraemer/PECUZAL_python.git",
    #packages=setuptools.find_packages(exclude=['docs', 'tests*']),
    packages=setuptools.find_packages(exclude=['docs']),
    py_modules=["pecuzal_embedding"],
    package_dir={'':'src'},
    package_data={'': ['src/data/*.csv']},
    include_package_data=True,
    data_files=[('data', ['./data/lorenz_pecora_multi.csv','./data/lorenz_pecora_uni_x.csv','./data/roessler_test_series.csv'])],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable"],
    python_requires='>=3.7',
    install_requires=["numpy>=1.17.2","scipy>=1.3.1,<1.9","scikit-learn>=0.21.3","progress>=1.5"], 
)

