import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchnmf",
    version="0.0.1",
    author="Chin Yun Yu",
    author_email="lolimaster.cs03@nctu.edu.tw",
    description="A pytorch package for Non-negative Matrix Factorization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yoyololicon/pytorch-NMFs",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)