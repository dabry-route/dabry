# Trajectory optimization in flow fields

NOTE: The article "Schnitzler et al., 
General extremal field method for time-optimal trajectory planning in flow fields, DOI 10.1109/LCSYS.2023.3284339"
refers to release v1.0.0 of the source code.

![](docs/movor.gif)

## Presentation

This module tackles trajectory optimization problems in strong,
non-uniform and unsteady flow fields, in particular wind fields.

The module performs origin-to-destination trajectory optimization (wrt. time)
by sampling extremal trajectories.
Extremal trajectories are Hamiltonian-minimizing trajectories.
The time-optimal trajectory of a problem, when it exists,
is a particular extremal trajectory.
So the set of all extremal trajectories (parametrized by a real value, 
the initial angle $\theta_0 \in [0, 2\pi[$) is guaranteed to contain the 
time-optimal trajectory, and this motivates the extremal integration method.

The module provides the following features for flow fields:
- Python code for analytic flow fields (see `dabry/flowfield.py`)
- Custom numpy zip (.npz) format definition for discrete flow fields (see `docs/flow_format.md`)
  - Translation from GRIB2 files to the npz format

A demonstration notebook is given at `examples/Gyre.ipynb` and can be
viewed using [nbviewer.org](https://nbviewer.org).

The module supports 2D planar environment as well as spherical problems.

## Cloning the repo

Clone the module using
```sh
git clone [repo-url]
```

## Installation as Python module 

After cloning the repo using the previous command,
install the project as a Python module using
```sh
python3 -m pip install -e ./dabry
```

### Base examples

You can now run base examples calling `dabry` as Python module:
```shell
python3 -m dabry case [case_name]
```
Results will be automatically saved to a new folder 
with the case's name in the current working directory.
The different output files format specification can be found in `docs`.

Available default problems for `case_name` 
are listed in `dabry/problems.csv`

### Real data

To directly run trajectory optimization on real wind fields, 
you have to install the `cdsapi` module.
```shell
python3 -m pip install cdsapi
```

Then, configure your [CDS Python API key](https://cds.climate.copernicus.eu/api-how-to)
for the `cdsapi` module to be allowed to extract wind fields from the
CDS database.

After that you can run trajectory optimization on real problems using

```shell
python3 -m dabry real [lon_start] [lat_start] [lon_target] [lat_target] [date_start] [airspeed] [altitude]
```

Note that an automatic CLI generator for `dabry`'s real cases
is available on the 
[website](https://dabry-navigation.github.io/real_data/).

## Visualization

Make sure the dependencies from `requirements_display.txt` are installed.

If the previous computation put the results in the `"movor (scaled)"` directory (for instance),
then the interactive display can be launched using
```shell
python -m dabry.display "movor (scaled)"
```

If `easygui` is installed, then the command

```shell
python -m dabry.display .
```
launches an interactive prompt to select the example to display from 
current directory.