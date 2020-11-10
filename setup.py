import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pecuzal_embedding", 
    version="0.0.1",
    author="K.H.Kraemer",
    author_email="hkraemer@pik-potsdam.de",
    description="PECUZAL automatic embedding of uni- and multivariate time series",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://gitlab.com/hkraemer2/pecuzal-python",
    packages=setuptools.find_packages(),
    py_modules=["pecuzal_embedding"],
    package_dir={'':'src'},
    include_package_data=True
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires='>=3.6',
    install_requires=["math","numpy>=1.17.2","scipy>=1.3.1","random","scikit-learn>=0.21.3"] 
    extras_requires = {"dev": [
        'unittest'
    ]}
)

