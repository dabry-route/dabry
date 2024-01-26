# Trajectory optimization in flow fields

![](docs/movor.gif)

## Presentation

This module tackles trajectory optimization problems in strong,
non-uniform and unsteady flow fields, in particular wind fields.

It provides the following features for wind fields:
- Code-level classes of analytic wind fields
- Custom H5 format definition for wind grid data
  - Translation from Windy API data to custom format
  - Translation from grib format to custom format
 
The module performs point-to-point trajectory optimization, which can 
be done using:
- A custom extremal shooting, Python coded algorithm
- The Matlab front tracking module [ToolboxLS](https://www.cs.ubc.ca/~mitchell/ToolboxLS/)

The module supports 2D planar environment as well as spherical problems.

## Installation (basic usage)

Open an appropriate working directory and clone this repo using `git clone [repo-url]`.

### Virtual environment

A virtual environement is recommended to install the module.
If you don't want to use it, go to next section.

To create a virtual environment, run
```sh
python3 -m venv env
```
Then activate the environment using
```sh
source env/bin/activate
```

### Module

Install the project as a Python module using
```sh
python3 -m pip install ./dabry
```

### Configuration

For `dabry` to run properly, the variable `DABRYPATH` shall point to the cloned repo root.
The following command
```sh
export DABRYPATH=[path_to_cloned_repo]
```
shall be run prior to using the module.

You can add it and the end of the `env/bin/activate` script,
using for instance:

```shell
echo "export DABRYPATH=[path_to_cloned_repo]" >> env/bin/activate
```

You can equivalently add it to your `~/.bashrc` script.

## Running the solver

### Base examples

You can now run base examples calling `dabry` as Python module:
```shell
python3 -m dabry case [case-name]
```
Results will be automatically saved to the `output` directory 
in the `dabry` root folder.
The different output files are
saved in `.h5` custom formats, which specification can be found in `docs`.

Check out [Dabry website](https://dabry-navigation.github.io/) to see available examples.

### Real data

You can also run trajectory optimization on real problems using

```shell
python3 -m dabry real [lon-start] [lat-start] [lon-target] [lat-target] [date-start] [airspeed] [altitude]
```

For this you need an access to the [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu)
for wind data extraction.

The `cdsapi` Python module is already installed but 
you still need to install a [CDS Python API key](https://cds.climate.copernicus.eu/api-how-to).

Note that an automatic CLI generator for `dabry`'s real cases
is available on the 
[website](https://dabry-navigation.github.io/real_data/).

## Installation (development)

For development purposes, you only need to clone the repo and set the variables:

 - `DABRYPATH` shall point to the cloned repo
 - `PYTHONPATH` shall point to the `src` directory within repo

1) If you want to run the Python scripts from the shell, you can set the previous
variables automatically by `cd` to repo root and using
```sh
source ./activate
```
2) If you run from IDE, make sure the variables are set appropriately


## Visualization

Visualize results using the visualisation module `dabry-visu` (https://github.com/bschnitzler/dabry-visu)
