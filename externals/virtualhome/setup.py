import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="virtualhome",
    version="2.3.0",
    author="Xavier Puig",
    author_email="xavierpuig@csail.mit.edu",
    description="Python API to communicate with the VirtualHome environment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xavierpuigf/virtualhome",
    project_urls={
        "Documentation": "http://virtual-home.org/docs/",
        "Bug Tracker": "https://github.com/xavierpuigf/virtualhome/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
)

