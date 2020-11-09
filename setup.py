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
    long_description_content_type="text/markdown",
    url="https://gitlab.com/hkraemer2/pecuzal-python",
    packages=setuptools.find_packages(),
    py_modules=["pecuzal_embedding"],
    package_dir={'':'src'},
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

