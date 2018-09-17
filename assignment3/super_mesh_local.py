from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import seaborn


# For "allclose" comparisions.
EPS = 1e-8


class LocalProblem(object):

    def __init__(self, t_donor, points, T_target, EN_target):
        self.t_donor = t_donor
        self.points = points
        self.T_target = T_target
        # Edge neighbors (-1 corresponds to a boundary edge)
        self.EN_target = EN_target
        # A local workspace for computng barycentric coordinates,
        # to avoid re-allocating a small matrix multiple times.
        self.local_mat = np.ones((3, 3))

    def plot(self, block=True):
        plt.plot(self.t_donor[:, 0], self.t_donor[:, 1],
                 marker='o', linestyle='None', color='black')
        plt.triplot(self.points[:, 0], self.points[:, 1],
                    self.T_target)
        centroids = np.mean(self.points[self.T_target, :], axis=1)
        for i, centroid in enumerate(centroids):
            plt.text(centroid[0], centroid[1], 't%d' % (i,),
                     horizontalalignment='center',
                     verticalalignment='center')
        for i, point in enumerate(self.points):
            plt.text(point[0], point[1], 'n%d' % (i,))
        plt.axis('scaled')
        plt.show(block=block)

    def barycentric_mat(self, tri):
        self.local_mat[:, 1:] = self.points[tri, :]
        return np.linalg.inv(self.local_mat)

    def barycentric_coords(self, tri, to_convert, bary_mat=None):
        if bary_mat is None:
            bary_mat = self.barycentric_mat(tri)
        return (bary_mat[[0], :] +
                to_convert.dot(bary_mat[1:, :]))

    @staticmethod
    def classify_intersections(indices, barycentric_coords):
        # Summing the number of "True" gives a count of the
        # number of barycentric coordinates "equal" to 0.
        nnz = np.sum(np.abs(barycentric_coords[indices, :]) < EPS,
                     axis=1)
        return nnz

    def find_containing(self):
        """Find a target triangle that intersects the donor triangle.

        Returns the index of the first target triangle containing
        any one of the nodes of the donor triangle.
        """
        for j, tri in enumerate(self.T_target):
            barycentric_coords = self.barycentric_coords(
                tri, self.t_donor)
            # We need every barycenter value to be >= 0, which forces
            # the point to be interior (since they sum to 1, all
            # positive guarantees all <= 1). We take all() across
            # rows since by construction the rows contain the
            # barycenter coordinates for each row in ``t_donor``.
            is_interior = np.all(barycentric_coords >= -EPS, axis=1)
            indices, = np.where(is_interior)
            if len(indices) > 0:
                classified = self.classify_intersections(
                    indices, barycentric_coords)
                return j, indices, classified
        # If we reach this point, no match was found.
        raise ValueError('No match found')


def main():
    t_donor = np.array([
        [0., -0.04895980747305631],  # WAS 0: [0., 0.],
        [1., -0.1859997917819376],  # WAS 1: [1., 0.],
        [0.46081253651398035, 1.],  # WAS 2: [0., 1.],
    ])
    T_target = np.array([
        [0, 6, 3],
        [6, 5, 3],
        [1, 6, 0],
        [6, 1, 4],
        [6, 2, 5],
        [2, 6, 4],
    ], dtype=np.int32)
    EN_target = np.array([
        [2, 1, -1],
        [4, -1, 0],
        [3, 0, -1],
        [2, -1, 5],
        [5, -1, 1],
        [4, 3, -1],
    ], dtype=np.int32)

    points = np.array([
        [-0.1298005199879263, -0.89006395817213257],
        # WAS 1: [1.4670185022336919, -0.032384407513456726],
        [1.4670185022336919, -0.25],
        [0.36031783199216916, 1.4810250468061925],
        [-0.64301200855208374, 0.039158548089343492],
        [1.2782442105358984, 0.67524939313982868],
        [-0.63761368148142072, 0.93883300293395822],
        # WAS 6: [0.27613335339321132, -0.21948481104925055],
        [0.7, -0.14488779648927319],
    ])
    loc_prob = LocalProblem(t_donor, points, T_target, EN_target)
    loc_prob.plot(block=False)
    target_match = loc_prob.find_containing()
    print(target_match)
    return loc_prob


LL = main()
B = np.array([[0., 0.], [3., 0.], [3., 4.]])
# C = np.array([[1.5, 0.], [3., 2.], [1.5, 2.]])
C = np.array([[1.5, -0.5], [4., 2.], [1., 3.]])
# Delta_v
dv = B[[1, 2, 0], :] - B
# n_{Delta_v}
dv /= np.linalg.norm(dv, ord=2, axis=1)[:, np.newaxis]
# dv_v = np.sum(dv * B, axis=1)[:, np.newaxis]
# proj = (dv[:, [0, 0, 1, 1]] * dv[:, [0, 1, 0, 1]]).reshape(
#     (6, 2), order='C')
# u_minus_v[i, j, :] = uj - vi = C[j, :] - B[i, :]
u_minus_v = C[np.newaxis, :, :] - B[:, np.newaxis, :]
# First three rows are uj - v0, then uj - v1, then uj - v2
u_minus_v = u_minus_v.reshape((9, 2), order='C')
# Take dot product manually to find coeff of projection.
dv = dv[[0, 0, 0, 1, 1, 1, 2, 2, 2], :]
kappa = np.sum(dv * u_minus_v, axis=1)[:, np.newaxis]
# First three rows are proj(uj) onto first edge,
# second three rows are proj(uj) onto second edge, etc.
proj = B[[0, 0, 0, 1, 1, 1, 2, 2, 2], :] + dv * kappa
# Find the signed length of the perpendicular parts
dv_perp = dv[:, [1, 0]]
dv_perp[:, 0] *= -1
sgn_len = np.sum(dv_perp * u_minus_v, axis=1)[:, np.newaxis]
lb = sgn_len[[1, 2, 0, 4, 5, 3, 7, 8, 6], :]
weight = lb / (lb - sgn_len)
pb = proj[[1, 2, 0, 4, 5, 3, 7, 8, 6], :]
intersections = proj * weight + pb * (1 - weight)
plt.figure()
plt.plot([0, 3, 3, 0], [0, 0, 4, 0])
plt.plot(C[:, 0], C[:, 1], color='black',
         marker='o', linestyle='None')
plt.plot(intersections[:, 0], intersections[:, 1], color='green',
         marker='o', linestyle='None')
plt.axis('scaled')
#plt.xlim(-0.5, 5)
#plt.ylim(-1, 4.5)
plt.show()
import matplotlib.path
pB = matplotlib.path.Path(B)
pC = matplotlib.path.Path(C)
