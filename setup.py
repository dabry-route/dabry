import os
import setuptools

# Code adapted from https://github.com/StanfordASL/hj_reachability/setup.py

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_version():
    with open(os.path.join(_CURRENT_DIR, "dabry", "__init__.py")) as f:
        for line in f:
            if line.startswith("__version__") and "=" in line:
                version = line[line.find("=") + 1:].strip(" '\"\n")
                if version:
                    return version
        raise ValueError("`__version__` not defined in `dabry/__init__.py`")


def _parse_requirements(file):
    with open(os.path.join(_CURRENT_DIR, file)) as f:
        return [line.rstrip() for line in f if not (line.isspace() or line.startswith("#"))]


setuptools.setup(
    name="dabry",
    version=_get_version(),
    description="Zermelo's problem resolution using extremal trajectories",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Bastien Schnitzler",
    author_email="bastien.schnitzler@live.fr",
    url="https://github.com/dabry-route/dabry",
    license="GPL-3",
    packages=setuptools.find_packages(),
    install_requires=_parse_requirements("requirements.txt"),
    include_package_data=True,
    package_data={'': ['problems.csv', 'data/*']}
)
