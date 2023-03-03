# Optimal trajectory planning in winds

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
- An external front tracking, Matlab coded module (ToolboxLS)

The module supports 2D planar environment as well as spherical problems.

## Installation

Clone this repo and open a shell in the directory where it has been cloned

### Virtual environment

Create a new virtual environment
```sh
python3 -m venv env
```
Activate it
```sh
source env/bin/activate
```
Now install dependencies
```sh
python3 -m pip install -r requirements.txt
```

### Configuration

The variable `DABRYPATH` shall point to the cloned repo root.
The variable `PYTHONPATH` shall also point to the `src` folder for Python to find the module.

1) If you run the scripts from the shell, first load paths using
```sh
source ./activate
```
2) If you run from IDE, make sure the variables are set appropriately


## Running the solver

You can run the demo script `examples/double_gyre.py` which provides an example
of how to run the solver.

Results are saved to the `output` folder. The different output files are
saved in `.h5` custom formats, which specification can be found in `docs`

### Visualization

Visualize results using the visualisation module `dabry-visu` (https://github.com/bschnitzler/dabry-visu)
