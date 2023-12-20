import random
import numpy as np

from dabry.dynamics import Utils
from dabry.problem import DatabaseProblem
from dabry.misc import Utils


if __name__ == '__main__':
    pb = DatabaseProblem('/home/bastien/Documents/data/wind/ncdc/tmp.mz/wind.h5')
    bl = np.array((pb.model.ff.grid[:, :, 0].min(), pb.model.ff.grid[:, :, 1].min()))
    tr = np.array((pb.model.ff.grid[:, :, 0].max(), pb.model.ff.grid[:, :, 1].max()))

    t = 0.5 * (pb.model.ff.t_end + pb.model.ff.t_start)
    psi = 0.3
    for _ in range(10):
        point = np.array(((tr[0] - bl[0]) * random.random() + bl[0],
                          (tr[1] - bl[1]) * random.random() + bl[1]))
        def mdiff(f, x, dx=1e-6):
            n = x.shape[0]
            vdx = np.diag(n * (1,)) * dx
            return np.column_stack(((f(x + vdx[i]) - f(x)) / vdx[i, i] for i in range(n)))
        findiff = mdiff(lambda x: pb.model.dyn.value(t, x, psi), point)
        anadiff = pb.model.dyn.d_value__d_state(t, point, psi, )
        b = findiff - anadiff
        pass
