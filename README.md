# Trajectory optimization in flow fields

![](docs/movor.gif)

## Presentation

This module tackles trajectory optimization problems in strong,
non-uniform and unsteady flow fields, in particular wind fields.

It provides the following features for flow fields:
- Python code for analytic flow fields (see `dabry/flowfield.py`)
- Custom numpy zip (.npz) format definition for discrete flow fields (see `docs/flow_format.md`)
  - Translation from GRIB2 files to the npz format
 
The module performs origin-to-destination trajectory optimization (wrt. time)
by sampling extremal trajectories i.e. Hamiltonian-minimizing trajectories which
are guaranteed to contain the problem's solution when it exists.

A demonstration notebook is given at `examples/Gyre.ipynb` and can be
viewed using [nbviewer.org](https://nbviewer.org).

The module supports 2D planar environment as well as spherical problems.

## Installation (module mode)

Clone the module
```sh
git clone [repo-url]
```

Install the project as a Python module using
```sh
python3 -m pip install -e ./dabry
```

### Base examples

You can now run base examples calling `dabry` as Python module:
```shell
python3 -m dabry case [case-name]
```
Results will be automatically saved to a new folder 
with the case's name in the current working directory.
The different output files format specification can be found in `docs`.

Available default problems are listed in `dabry/problems.csv`

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

## Visualization

Visualize results using the visualisation module `dabry/display`
