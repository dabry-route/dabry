## Optimal trajectory planning in winds

### Presentation

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

### Getting started

1) Set the environment variable `MERMOZ_PATH` to the installation path
2) Checkout `examples/solver_usage.py` to perform problem solving

Results are saved to the `output` folder. The different output files are
saved in `.h5` custom formats, which specification can be found in `docs`