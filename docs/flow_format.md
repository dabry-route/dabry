# Flow format

Flow fields are saved to disk as a discrete representation.
This representation is meant to be a uniform sample of flow field values over time and space, in cartesian coordinates or lon/lat gcs coordinates depending on the computation mode.
A flow field discrete representation can be saved to disk as a numpy ZIP (.npz) containing the numpy arrays:

 - values.npy 
    - (nx/nlon, ny/nlat, 2) float if steady
    - (nt, nx/nlon, ny/nlat, 2) float if unsteady
 - bounds.npy (3, 2) float (time, space coord 1, space coord 2)
 - coords.npy (), string, either 'cartesian' or 'gcs'

Optional fields are:

 - grad_values.npy
    - (nx/nlon, ny/nlat, 2) float if steady
    - (nt, nx/nlon, ny/nlat, 2) float if unsteady

## Units

### Cartesian
The length units are named u_length (m, ft, ...) and the time units are named u_time (s, h, ...).

 - Magnitude: must be in u_length / u_time
 - Bounds:
    - Time bounds in u_time
    - Space bounds in u_length

### GCS data
When data comes from extraction of longitude/latitude data, units are of different nature as regards magnitude of the flow field vectors and the time and space coordinates. In this case, exact units are required, which are the following ones

 - Magnitude: meters per seconds
 - Bounds:
    - Time: seconds
    - Space: radians. Values are expected to be in ]-pi, pi].

