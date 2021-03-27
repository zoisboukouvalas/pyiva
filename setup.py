import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="pyiva",
    version="0.0.1",
    author="Zois Boukouvalas",
    author_email="boukouva@american.edu",
    description="Python implementation of independent vector analysis (IVA) with Laplacian prior",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zoisboukouvalas/pyiva",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
