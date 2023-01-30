import numpy as np


class LevelSet:

    def __init__(self, points, normals):
        self.points = np.zeros(points.shape)
        self.points[:] = points
        self.normals = np.zeros(normals.shape)
        self.normals[:] = normals

    def trajify(self, smoothing=1):
        """
        Return a fictitious trajectory going around the set
        :param smoothing: Degree of polynomial approximation
        :return: a trajectory
        """
        pass
