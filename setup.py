import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pybackpack",
    version="0.1.0",
    author="Pooya Vahidi",
    description="A collection of utils, helpers and tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pooyavahidi/pybackpack",
    packages=setuptools.find_packages(exclude=['tests*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'python-dateutil>=2.8.1'
    ],
    python_requires='>=3.6',
)