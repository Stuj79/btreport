import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="btreport",
    version="0.0.17",
    author="Stuart Jamieson",
    author_email="stuj79@hotmail.com",
    description="A module to help visualise and analyse the results of a bt module backtest",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Stuj79/btreport",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)