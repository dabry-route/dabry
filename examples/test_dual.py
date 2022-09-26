import numpy as np

from mermoz.feedback import ConstantFB
from mermoz.mdf_manager import MDFmanager
from mermoz.params_summary import ParamsSummary
from mermoz.problem import MermozProblem
from mermoz.model import ZermeloGeneralModel
from mermoz.stoppingcond import TimedSC
from mermoz.wind import UniformWind, VortexWind, RadialGaussWind, LCWind

if __name__ == '__main__':
    mdfm = MDFmanager()
    output_dir = f'/home/bastien/Documents/work/mermoz/output/example_test_dual'
    mdfm.set_output_dir(output_dir)
    mdfm.clean_output_dir()
    v_a = 23.
    sf = 1e6
    x_init = sf * np.array((0., 0.))
    x_target = sf * np.array((1., 0.))
    model = ZermeloGeneralModel(v_a)
    wind = LCWind(np.array((1., 1.)),
                  (UniformWind(np.array((1., 1.))), RadialGaussWind(sf * 0.5, sf * 0., sf * 0.1, 0.2, v_a * 3)))
    model.update_wind(wind.dualize())
    mdfm.dump_wind(wind, nx=50, ny=50, bl=sf * np.array((-0.2, -0.5)), tr=sf * np.array((1.2, 0.5)))

    mp = MermozProblem(model, x_init, x_target, 'cartesian')

    for u in np.linspace(-np.pi, np.pi, 50):
        mp.load_feedback(ConstantFB(u))
        sc = TimedSC(sf / v_a)
        mp.integrate_trajectory(mp.x_init, sc, max_iter=3000, int_step=10.)

    mdfm.dump_trajs(mp.trajs)
    ps = ParamsSummary()
    ps.set_output_dir(output_dir)
    ps.load_from_problem(mp)
    ps.dump()
