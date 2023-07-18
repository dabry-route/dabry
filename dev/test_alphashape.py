#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from datetime import datetime
import h5py

from dabry.problem import DatabaseProblem
from dabry.ddf_manager import DDFmanager
from dabry.misc import Utils, Chrono
from dabry.solver_ef import SolverEF
from dabry.penalty import CirclePenalty, DiscretePenalty


# In[2]:


penalty = DiscretePenalty()
penalty.load('test.h5')
penalty.d_value(datetime(2020, 7, 14, 0, 0).timestamp(), Utils.DEG_TO_RAD * np.array((-5, 10)))


# In[3]:


x_init = Utils.DEG_TO_RAD * np.array((-17.46, 14.71)) # Dakar
x_target = Utils.DEG_TO_RAD * np.array((-35.26, -5.81)) # Natal
start_date = datetime(2020, 7, 15, 0, 0)
airspeed = 23
level = '1000'

duration = 2 * Utils.distance(x_init, x_target,
                              coords=Utils.COORD_GCS) / airspeed
stop_date = datetime.fromtimestamp(start_date.timestamp() + duration)

ddf = DDFmanager()
ddf.setup()

ddf.retrieve_wind(start_date, stop_date, level=level, res='0.5')
case_name = ddf.format_cname(x_init, x_target, start_date.timestamp())

cache_wind = False
cache_rff = False

# This instance prints absolute elapsed time between operations
chrono = Chrono()

# Create a file manager to dump problem data
mdfm = DDFmanager(cache_wind=cache_wind, cache_rff=cache_rff)
mdfm.setup()
case_name = f'test_penalty_real'
mdfm.set_case(case_name)
mdfm.clean_output_dir()

# Space and time discretization
# Will be used to save wind when wind is analytical and shall be sampled
# Will also be used by front tracking module
nx_rft = 101
ny_rft = 101
nt_rft = 20

pb = DatabaseProblem(x_init=x_init,
                     x_target=x_target, airspeed=airspeed,
                     t_start=start_date.timestamp(), t_end=stop_date.timestamp(),
                     altitude=level,
                     resolution='0.5')

pb.penalty = DiscretePenalty(0.002 * penalty.data, penalty.ts, penalty.grid)
#with h5py.File('data/cds/0.5/1000/20210930.grib2', 'r') as f:
#    penalty = DiscretePenalty()

# pb.flatten()

if not cache_wind:
    chrono.start('Dumping windfield to file')
    mdfm.dump_wind(pb.model.wind, nx=nx_rft, ny=ny_rft, nt=nt_rft, bl=pb.bl, tr=pb.tr)
    chrono.stop()

# Setting the extremal solver
solver_ef = SolverEF(pb, pb.time_scale, max_steps=700, rel_nb_ceil=0.01, quick_solve=True)

chrono.start('Solving problem using extremal field (EF)')
res_ef = solver_ef.solve()
chrono.stop()
if res_ef.status:
    # Solution found
    # Save optimal trajectory
    mdfm.dump_trajs([res_ef.traj])
    print(f'Target reached in : {Utils.time_fmt(res_ef.duration)}')
else:
    print('No solution found')

# Save extremal field for display purposes
extremals = solver_ef.get_trajs()
mdfm.dump_trajs(extremals)

'''
pb.penalty = DiscretePenalty(0.001 * penalty.data, penalty.ts, penalty.grid)
solver_ef = SolverEF(pb, pb.time_scale, max_steps=700, rel_nb_ceil=0.02, quick_solve=True)
chrono.start('Solving problem using extremal field (EF)')
res_ef = solver_ef.solve()
chrono.stop()
if res_ef.status:
    # Solution found
    # Save optimal trajectory
    mdfm.dump_trajs([res_ef.traj])
    print(f'Target reached in : {Utils.time_fmt(res_ef.duration)}')
else:
    print('No solution found')
'''

pb.orthodromic()
mdfm.dump_trajs([pb.trajs[-1]])
mdfm.dump_penalty(pb.penalty)
mdfm.log(pb)
