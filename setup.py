"""Setup script for pymultipleis."""

import os
from setuptools import find_packages
from setuptools import setup


folder = os.path.dirname(__file__)


req_path = os.path.join(folder, "requirements.txt")
install_requires = []
if os.path.exists(req_path):
    with open(req_path) as fp:
        install_requires = [line.strip() for line in fp]

readme_path = os.path.join(folder, "README.md")
readme_contents = ""
if os.path.exists(readme_path):
    with open(readme_path, encoding='utf-8') as fp:
        readme_contents = fp.read().strip()

setup(
    name="pymultipleis",
    version="0.1.2",
    description="A library for fitting a sequence of electrochemical impedance spectra.",
    author="Richard Chukwu",
    author_email="richinex@gmail.com",
    url="https://github.com/richinex/pymultipleis",
    long_description=readme_contents,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    package_data={},
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    keywords="electrochemical impedance spectroscopy, batch fitting, multiple, jax",
    requires_python=">=3.9",
)
