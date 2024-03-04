# Flow format

Flow fields are saved to disk in a discrete representation.
This representation is meant to be a uniform sample of flow field values over time and space, in cartesian coordinates or lon/lat gcs coordinates depending on the computation mode.
A flow field discrete representation can be saved to disk as a numpy ZIP (.npz) containing the numpy arrays:

 - values.npy 
    - (nx/nlon, ny/nlat, 2) float if steady
    - (nt, nx/nlon, ny/nlat, 2) float if unsteady
 - bounds.npy
   - (3, 2) float if unsteady
   - (2, 2) float if steady
 - coords.npy (), string, either 'cartesian' or 'gcs'

The bounds are given in format
```
[
   [time_lower_bound, time_upper_bound],
   [space1_lower_bound, space1_upper_bound],
   [space2_lower_bound, space2_upper_bound]
]
```
with the time bounds absent if flow field is steady.

The underlying grid is understood as being an uniform grid in space and time,
with uniform sampling e.g. for time
```
np.linspace(bounds[0, 0], bounds[0, 1], values.shape[0])
```

Optional fields are:

 - grad_values.npy
    - (nx/nlon, ny/nlat, 2) float if steady
    - (nt, nx/nlon, ny/nlat, 2) float if unsteady

## Units

### Cartesian
Unit do not matter in cartesian mode as long as they are consistent together.

If the lengths are in u_length (m, ft, ...) and the time is in u_time (s, h, ...), then

 - Flow field magnitude: must be in u_length / u_time
 - Bounds:
    - Time bounds must be in u_time
    - Space bounds must be in u_length

### GCS data
When data comes from extraction of longitude/latitude data, units are of different nature as regards magnitude of the flow field vectors and the time and space coordinates. In this case, exact units are required, which are the following ones

 - Flow field magnitude: meters per seconds
 - Bounds:
    - Time: seconds
    - Space: radians. Values are expected to be in ]-pi, pi].

Note that most data source provide longitude and latitude in degrees
and consequently should be preprocessed.