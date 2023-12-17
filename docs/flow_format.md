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

Units are SI.

 - For the flow field magnitude, meters per second
 - For time bounds, seconds
 - For the space bounds, 
    - meters in cartesian mode
    - radians in gcs mode. Values are expected to be in ]-pi, pi].

