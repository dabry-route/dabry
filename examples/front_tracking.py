import time

import numpy as np
import os

from mermoz.problem import IndexedProblem, problems
from mermoz.rft import RFT
from mermoz.shooting import Shooting
from mermoz.mdf_manager import MDFmanager
from mermoz.params_summary import ParamsSummary

if __name__ == '__main__':
    # Choose problem ID
    pb_id = 2

    run_rft = False
    run_pmp = True

    output_dir = f'/home/bastien/Documents/work/mermoz/output/example_ft_{problems[pb_id][1]}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Create a file manager to dump problem data
    mdfm = MDFmanager()
    mdfm.set_output_dir(output_dir)
    mdfm.clean_output_dir()

    pb = IndexedProblem(pb_id, seed=2)
    print(pb.bl)

    mdfm.dump_wind(pb.model.wind, nx=51, ny=51, bl=pb.bl, tr=pb.tr)

    T = 1.1 * pb._geod_l / pb.model.v_a

    ps = ParamsSummary()
    ps.set_output_dir(output_dir)
    ps.load_from_problem(pb)
    ps.add_param('max_time', T)

    if run_rft:
        # Setting front tracking algorithm
        nx_rft = 51
        ny_rft = 51
        nt_rft = 10

        delta_x = (pb.tr[0] - pb.bl[0]) / (nx_rft - 1)
        delta_y = (pb.tr[1] - pb.bl[1]) / (ny_rft - 1)

        print(f"Tracking reachability front ({nx_rft}x{ny_rft}, dx={delta_x:.2E}, dy={delta_y:.2E})... ", end='')
        t_start = time.time()

        rft = RFT(pb.bl, pb.tr, T, nx_rft, ny_rft, nt_rft, pb, pb.x_init, kernel='matlab', coords=pb.coords)

        rft.compute()

        t_end = time.time()
        rft_time = t_end - t_start
        print(f"Done ({rft_time:.3f} s)")

        rft.dump_rff(output_dir)

        k = 0
        while True:
            if rft.value(pb.x_target, k) < 0.:
                nt_rft_eff = k
                break
            k += 1

        ps.add_param('nt_rft', nt_rft)
        ps.add_param('nx_rft', nx_rft)
        ps.add_param('ny_rft', ny_rft)
        ps.add_param('rft_time', rft_time)
        ps.add_param('nt_rft_eff', nt_rft_eff)

    if run_pmp:
        # Set a list of initial adjoint states for the shooting method
        nt_pmp = 100
        initial_headings = np.linspace(0.1, 2 * np.pi - 0.1, 30)
        list_p = list(map(lambda theta: -np.array([np.cos(theta), np.sin(theta)]), initial_headings))

        print(f"Shooting PMP trajectories ({len(list_p)})... ", end='')
        t_start = time.time()

        for k, p in enumerate(list_p):
            shoot = Shooting(pb.model.dyn, pb.x_init, T, adapt_ts=False, N_iter=nt_pmp, domain=pb.domain,
                             coords=pb.coords)
            shoot.set_adjoint(p)
            aug_traj = shoot.integrate()
            pb.trajs.append(aug_traj)

        t_end = time.time()
        pmp_time = t_end - t_start
        print(f"Done ({pmp_time:.3f} s)")

        mdfm.dump_trajs(pb.trajs)

        ps.add_param('nt_pmp', nt_pmp)
        ps.add_param('pmp_time', pmp_time)

    ps.dump()
