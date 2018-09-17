from __future__ import print_function

import collections
import itertools
import time

import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial
import six

Q = []  # XXX TEMP
M_MAT = np.array([
    [2.0, 1.0, 1.0],
    [1.0, 2.0, 1.0],
    [1.0, 1.0, 2.0],
]) / 12.0
# For "allclose" comparisions.
EPS = 1e-8


def in_polygon(points, boundary):
    polygon = matplotlib.path.Path(boundary)
    return polygon.contains_points(points)


def unique_points(points):
    # Assumes points is N x 2.
    to_keep = np.zeros(points.shape)
    kept = 0
    for point in points:
        match = False
        for other_pt in to_keep[:kept, :]:
            if np.allclose(point, other_pt):
                match = True
                break
        if not match:
            to_keep[kept, :] = point
            kept += 1
    return to_keep[:kept, :]


def get_edge_map(triangles):
    edge_map = collections.defaultdict(list)
    for tri_ind, triangle in enumerate(triangles):
        n0, n1, n2 = sorted(triangle)
        edge_map[(n0, n1)].append(tri_ind)
        edge_map[(n0, n2)].append(tri_ind)
        edge_map[(n1, n2)].append(tri_ind)
    return dict(edge_map)


def line_intersect(v0, v1, u0, u1):
    # v(s) = v0 + s (v1 - v0)
    dv = v1 - v0
    # u(t) = u0 + t (u1 - u0)
    du = u1 - u0
    # v0 + s dv = u0 + t du
    # [dv -du][s, t] = u0 - v0
    M = np.zeros((2, 2))
    M[:, 0] = dv
    M[:, 1] = -du
    d0 = u0 - v0
    try:
        s, t = np.linalg.solve(M, d0)
    except np.linalg.LinAlgError:
        normal = du[::-1].copy()
        normal[0] *= -1
        normal /= np.linalg.norm(normal, ord=2)
        Cu = np.dot(normal, u0)
        Cv = np.dot(normal, v0)
        if np.allclose(Cu, Cv):
            p = normal[::-1].copy()
            p[1] *= -1
            # Write the line as p(0) + r p
            start_u, end_u = np.dot(p, u0), np.dot(p, u1)
            if start_u > end_u:
                start_u, end_u = end_u, start_u
                u0, u1 = u1, u0
            start_v, end_v = np.dot(p, v0), np.dot(p, v1)
            if start_v > end_v:
                start_v, end_v = end_v, start_v
                v0, v1 = v1, v0

            if np.allclose(start_v, end_u):
                return [u1]
            elif np.allclose(start_u, end_v):
                return [u0]
            if start_v > end_u or start_u > end_v:
                return
            else:
                if (np.allclose(start_v, start_u) and
                    np.allclose(end_v, end_u)):
                    return [u0, u1]
                else:
                    raise ValueError('Parallel lines intersect on '
                                     'an interval, hence non-unique')
        else:
            return

    if not (0 < s < 1 or np.allclose(s, 0) or np.allclose(s, 1)):
        return
    if not (0 < t < 1 or np.allclose(t, 0) or np.allclose(t, 1)):
        return
    return [v0 + s * dv]


class SimpleMesh(object):

    def __init__(self, n, dt):
        self.n = n
        self.dt = dt
        # A "local" matrix to be used for computing areas.
        self.local_mat = np.zeros((3, 3))
        self.local_mat[:, 0] = 1.0

        self.target_nodes, self.target_triangles = self.basic_mesh()
        self.donor_nodes, self.donor_triangles = self.relabel_mesh()
        self.ghost_map = self.make_ghost()
        self.rotate_target()

        self.target_edges = get_edge_map(self.target_triangles)
        self.donor_edges = get_edge_map(self.donor_triangles)
        self.intersections = self.get_edge_intersections()
        self.supermesh = self.get_supermesh()
        self.lhs_mat, self.rhs_mat = self.get_system()
        self.lhs_solve = scipy.sparse.linalg.splu(self.lhs_mat)
        self.solution = None

    def compute_area(self, tri):
        # Assumes column 0 of `local_mat` has not been disturbed.
        self.local_mat[:, 1:] = tri
        return 0.5 * np.linalg.det(self.local_mat)

    def get_p1_coeffs(self, tri):
        # Assumes column 0 of `local_mat` has not been disturbed.
        self.local_mat[:, 1:] = tri
        return np.linalg.inv(self.local_mat)

    def eval_p1(self, tri, coeffs):
        # Assumes column 0 of `local_mat` has not been disturbed.
        self.local_mat[:, 1:] = tri
        return self.local_mat.dot(coeffs)

    def add_solution(self, func):
        values = func(self.donor_nodes)
        soln = values[self.donor_triangles].T
        self.solution = soln.reshape((soln.size, 1), order='F')

    def update_solution(self):
        rhs_val = self.rhs_mat.dot(self.solution)
        self.solution = self.lhs_solve.solve(rhs_val)

    def basic_mesh(self):
        lim = 1 + 2.0 / (self.n - 1)
        n = self.n + 2
        entries = np.linspace(-lim, lim, n)
        nodes = np.zeros((n**2, 2))
        triangles = np.zeros((2 * (n - 1)**2, 3), dtype=np.int32)
        node_ind = 0
        tri_ind = 0
        for i in six.moves.xrange(n):
            y_val = entries[i]
            for j in six.moves.xrange(n):
                x_val = entries[j]
                nodes[node_ind, :] = x_val, y_val
                if i < n - 1 and j < n - 1:
                    triangles[tri_ind, :] = (node_ind, node_ind + 1,
                                             node_ind + n + 1)
                    tri_ind += 1
                    triangles[tri_ind, :] = (node_ind, node_ind + n + 1,
                                             node_ind + n)
                    tri_ind += 1
                node_ind += 1
        return nodes, triangles

    def relabel_mesh(self):
        """Relabel so that "interior" nodes and triangles come first.

        We intentionally built a ghost layer around our mesh and
        so peel off the layer.
        """
        nodes = self.target_nodes
        inside_bool = (
            np.all(nodes >= [-1 - EPS, -1 - EPS], axis=1) &
            np.all(nodes <= [1 + EPS, 1 + EPS], axis=1))
        inside_indices, = np.where(inside_bool)
        outside_indices, = np.where(~inside_bool)
        new_indices = np.hstack([inside_indices, outside_indices])

        tri_inside_bool = np.in1d(
            self.target_triangles.flatten(order='F'),
            inside_indices)
        tri_inside_bool = tri_inside_bool.reshape(
            self.target_triangles.shape, order='F')
        tri_inside_bool = np.all(tri_inside_bool, axis=1)
        inside_tris, = np.where(tri_inside_bool)
        outside_tris, = np.where(~tri_inside_bool)
        new_tri_indices = np.hstack([inside_tris, outside_tris])

        self.target_triangles = (
            self.target_triangles[new_tri_indices, :])
        index_remap = {curr: i for i, curr in enumerate(new_indices)}
        for tri in self.target_triangles:
            tri[0] = index_remap[tri[0]]
            tri[1] = index_remap[tri[1]]
            tri[2] = index_remap[tri[2]]
        self.target_nodes = self.target_nodes[new_indices, :]

        # Peel off internal-only.
        return (self.target_nodes[:len(inside_indices), :].copy(),
                self.target_triangles[:len(inside_tris), :].copy())

    def rotate_target(self):
        c = np.cos(self.dt)
        s = np.sin(self.dt)
        M = np.array([
            [c, s],
            [-s, c],
        ])
        self.target_nodes = self.target_nodes.dot(M)

    def make_ghost(self):
        num_inside, _ = self.donor_triangles.shape
        num_total, _ = self.target_triangles.shape
        ghost_map = {}
        for index in six.moves.xrange(num_inside, num_total):
            tri_inds = self.target_triangles[index, :]
            curr_t = self.target_nodes[tri_inds, :].copy()
            # If we're to the left of -1, move 2 units to right.
            min_x = np.min(curr_t[:, 0])
            if min_x < -1.0:
                curr_t[:, 0] += 2
            else:
                # If we're to the right of 1, move 2 units to left.
                max_x = np.max(curr_t[:, 0])
                if max_x > 1.0:
                    curr_t[:, 0] -= 2

            # If we're below -1, move 2 units up.
            min_y = np.min(curr_t[:, 1])
            if min_y < -1.0:
                curr_t[:, 1] += 2
            else:
                # If we're above 1, move 2 units down.
                max_y = np.max(curr_t[:, 1])
                if max_y > 1.0:
                    curr_t[:, 1] -= 2

            delta = (self.donor_nodes[:, np.newaxis, :] -
                     curr_t[np.newaxis, :, :])
            pairwise = np.linalg.norm(delta, ord=2, axis=2)
            nearest_rows = np.argmin(pairwise, axis=0)
            # Sanity check:
            # np.allclose(self.donor_nodes[nearest_rows, :], curr_t)
            (ghost_index,), = np.where(
                np.all(self.donor_triangles == nearest_rows, axis=1))
            ghost_map[index] = ghost_index

        return ghost_map

    def get_edge_intersections(self):
        result = {}
        for e_donor in six.iterkeys(self.donor_edges):
            u0 = self.donor_nodes[e_donor[0], :]
            u1 = self.donor_nodes[e_donor[1], :]
            for e_target in six.iterkeys(self.target_edges):
                key = (e_donor, e_target)
                v0 = self.target_nodes[e_target[0], :]
                v1 = self.target_nodes[e_target[1], :]
                points = line_intersect(v0, v1, u0, u1)
                if points is not None:
                    result[key] = points

        return result

    def get_supermesh(self):
        result = {}
        for i, t_donor in enumerate(self.donor_triangles):
            # We need ordered edges (for uniqueness).
            n0, n1, n2 = sorted(t_donor)
            edges_donor = ((n0, n1), (n0, n2), (n1, n2))
            ti = self.donor_nodes[t_donor, :]
            result[i] = curr_tri = {}
            for j, t_target in enumerate(self.target_triangles):
                n0, n1, n2 = sorted(t_target)
                edges_target = ((n0, n1), (n0, n2), (n1, n2))
                edge_pts = []
                for edge_pair in itertools.product(edges_donor,
                                                   edges_target):
                    if edge_pair in self.intersections:
                        edge_pts.extend(self.intersections[edge_pair])
                if edge_pts:
                    tj = self.target_nodes[t_target, :]
                    ti_nodes, = np.where(in_polygon(ti, tj))
                    for ti_node in ti_nodes:
                        edge_pts.append(ti[ti_node, :])
                    tj_nodes, = np.where(in_polygon(tj, ti))
                    for tj_node in tj_nodes:
                        edge_pts.append(tj[tj_node, :])

                    # Perform a Delaunay triangulation, since we'll be
                    # breaking up the intersection into triangles to
                    # be integrated over.
                    edge_pts = np.array(edge_pts)
                    edge_pts = unique_points(edge_pts)
                    num_pts, _ = edge_pts.shape
                    if (num_pts < 3 or
                        np.linalg.matrix_rank(edge_pts) < 2):
                        Q.append((i, j, edge_pts))  # XX
                    else:
                        curr_tri[j] = scipy.spatial.Delaunay(edge_pts)
        return result

    def add_rhs_coeffs(self, donor_ind, donor_coeffs, rhs_mat):
        curr_tri = self.supermesh[donor_ind]
        donor_slice = slice(3 * donor_ind, 3 * (donor_ind + 1))
        for target_ind, delaunay_tri in six.iteritems(curr_tri):
            # First determine if this element is a ghost element.
            if target_ind in self.ghost_map:
                true_target = self.ghost_map[target_ind]
            else:
                true_target = target_ind
            target_slice = slice(3 * true_target,
                                 3 * (true_target + 1))

            target_tri = self.target_triangles[target_ind, :]
            t_target = self.target_nodes[target_tri, :]
            target_coeffs = self.get_p1_coeffs(t_target)

            # Loop through each local triangle and perform
            # quadrature.
            for local_tri in delaunay_tri.simplices:
                t_local = delaunay_tri.points[local_tri, :]
                area_T = self.compute_area(t_local)
                centroid = np.mean(t_local, axis=0)
                quad_pts = (t_local + centroid) / 2.0
                # The rows of donor_phi corresponds to quadrature
                # points qi and the columns to PHI_j.
                donor_phi = self.eval_p1(quad_pts, donor_coeffs)
                # The rows of target_phi corresponds to qi and the
                # columns to PHI(hat)_j
                target_phi = self.eval_p1(quad_pts, target_coeffs)
                # By taking donor_phi.T and multiplying by target_phi,
                # the result matrix has dot products of columns in
                # each entry.
                quad_vals = donor_phi.T.dot(target_phi)
                quad_vals /= 3.0
                quad_vals *= area_T
                rhs_mat[donor_slice, target_slice] += quad_vals

    def get_system(self):
        num_tri, _ = self.donor_triangles.shape
        lhs_mat = scipy.sparse.dok_matrix((3 * num_tri, 3 * num_tri))
        rhs_mat = scipy.sparse.dok_matrix((3 * num_tri, 3 * num_tri))

        for donor_ind in six.iterkeys(self.supermesh):
            donor_tri = self.donor_triangles[donor_ind, :]
            t_donor = self.donor_nodes[donor_tri, :]
            area_T = self.compute_area(t_donor)

            sq_slice = slice(3 * donor_ind, 3 * (donor_ind + 1))
            lhs_mat[sq_slice, sq_slice] = M_MAT * area_T
            donor_coeffs = self.get_p1_coeffs(t_donor)
            self.add_rhs_coeffs(donor_ind, donor_coeffs, rhs_mat)

        return lhs_mat.tocsc(), rhs_mat.tocsc()

    def plot(self):
        plt.triplot(self.donor_nodes[:, 0], self.donor_nodes[:, 1],
                    self.donor_triangles)
        plt.plot(self.donor_nodes[:, 0], self.donor_nodes[:, 1], 'o')
        plt.triplot(self.target_nodes[:, 0], self.target_nodes[:, 1],
                    self.target_triangles)
        plt.plot(self.target_nodes[:, 0], self.target_nodes[:, 1], 'o')
        plt.axis('scaled')
        plt.show()

    def verify_ghost_map(self):
        plt.triplot(self.donor_nodes[:, 0], self.donor_nodes[:, 1],
                    self.donor_triangles)
        plt.plot(self.donor_nodes[:, 0], self.donor_nodes[:, 1], 'o')
        plt.triplot(self.target_nodes[:, 0], self.target_nodes[:, 1],
                    self.target_triangles)
        plt.plot(self.target_nodes[:, 0], self.target_nodes[:, 1], 'o')
        for i, tri in enumerate(self.target_triangles):
            centroid = np.mean(self.target_nodes[tri, :], axis=0)
            plt.text(centroid[0], centroid[1], str(i),
                     horizontalalignment='center',
                     verticalalignment='center')

        for pair in sorted(self.ghost_map.items()):
            print(pair)
        plt.axis('scaled')
        plt.show()

    def plot_overlay(self, donor_ind):
        t_donor = self.donor_triangles[donor_ind, :]
        ti = self.donor_nodes[t_donor, :]
        plt.plot(ti[[0, 1, 2, 0], 0], ti[[0, 1, 2, 0], 1],
                 color='blue')

        curr_tri = self.supermesh[donor_ind]
        sub_polys = []
        for target_ind, delaunay_tri in six.iteritems(curr_tri):
            t_target = self.target_triangles[target_ind, :]
            tj = self.target_nodes[t_target, :]
            sub_polys.append([tj, delaunay_tri])
            plt.plot(tj[[0, 1, 2, 0], 0], tj[[0, 1, 2, 0], 1],
                     color='red', linestyle='dashed')
            plt.scatter(delaunay_tri.points[:, 0],
                        delaunay_tri.points[:, 1], color='black')
        plt.axis('scaled')

        for tj, delaunay_tri in sub_polys:
            plt.figure()
            points = delaunay_tri.points
            tri_local = delaunay_tri.simplices
            plt.plot(ti[[0, 1, 2, 0], 0], ti[[0, 1, 2, 0], 1],
                     color='black', linestyle='dashed')
            plt.plot(tj[[0, 1, 2, 0], 0], tj[[0, 1, 2, 0], 1],
                     color='red', linestyle='dashed')
            plt.triplot(points[:, 0], points[:, 1], tri_local,
                        color='blue')
            plt.plot(points[:, 0], points[:, 1], 'o')
            plt.axis('scaled')

        plt.show()

    def plot_supermesh(self):
        # Just put a rough collection of nodes and triangles, and
        # allow duplicates in nodes.
        super_nodes = np.zeros((0, 2))
        super_tris = np.zeros((0, 3), dtype=np.int32)
        node_ind = 0
        for vals in six.itervalues(self.supermesh):
            for delaunay_tri in six.itervalues(vals):
                num_pts, _ = delaunay_tri.points.shape
                super_nodes = np.vstack(
                    [super_nodes, delaunay_tri.points.copy()])
                super_tris = np.vstack(
                    [super_tris,
                     delaunay_tri.simplices.copy() + node_ind])
                node_ind += num_pts

        colors = np.ones(super_tris.shape[0])
        plt.tripcolor(super_nodes[:, 0], super_nodes[:, 1],
                      super_tris, facecolors=colors,
                      alpha=0.2)
        plt.plot(super_nodes[:, 0], super_nodes[:, 1], 'o')
        plt.axis('scaled')
        plt.show()

    def to_plot_vals(self):
        """Hack to coerce DG values into same locations as nodes."""
        plot_vals = np.zeros(self.donor_nodes.shape[0])
        tri_index_base = 0
        for triangle in self.donor_triangles:
            for i in (0, 1, 2):
                plot_vals[triangle[i]] = (
                    self.solution[tri_index_base + i, :])
            tri_index_base += 3
        return plot_vals

    def plot_solution(self):
        plt.tricontourf(self.donor_nodes[:, 0], self.donor_nodes[:, 1],
                        self.donor_triangles, self.to_plot_vals(), 20,
                        cmap='viridis')
        plt.axis('scaled')
        plt.colorbar()
        plt.show()


def main1():
    mesh = SimpleMesh(4, 0.1)
    mesh.plot()


def main2():
    mesh = SimpleMesh(4, 0.1)
    mesh.verify_ghost_map()


def main3():
    mesh = SimpleMesh(4, 0.1)
    mesh.plot_overlay(1)


def main4():
    mesh = SimpleMesh(4, 0.2)
    mesh.plot_supermesh()


def periodic_soln(points):
    # sin(pi x) cos(pi y)
    return (np.sin(np.pi * points[:, 0]) *
            np.cos(np.pi * points[:, 1]))


def radial_soln(points):
    r = np.linalg.norm(points, ord=2, axis=1)
    abs_theta = np.abs(np.arctan2(points[:, 1], points[:, 0]))
    return np.maximum(0.75 - r, 0) * abs_theta


def main5():
    start = time.time()
    # 10 --> ~13s
    # 14 --> ~49s
    mesh = SimpleMesh(14, 0.05)
    duration = time.time() - start
    print('Duration: %g' % (duration,))
    mesh.add_solution(radial_soln)
    mesh.plot_solution()

    for _ in six.moves.xrange(4):
        for _ in six.moves.xrange(10):
            mesh.update_solution()
        mesh.plot_solution()
    return mesh
