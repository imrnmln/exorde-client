from setuptools import find_packages, setup

setup(
    name="exorde_data",
    version="0.0.1",
    packages=find_packages(include=["exorde_data"]),
    install_requires=["madtypes", "madframe"],
    entry_points={
        "console_scripts": ["exorde_data = exorde_data.__init__:print_schema"]
    },
    extras_require={
        "reddit": ["ap98j3envoubi3fco1kc"],
        "twitter": ["a7df32de3a60dfdb7a0b"],
        "dev": ["pytest"],
    },
)
