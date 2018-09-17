import matplotlib.pyplot as plt
import numpy as np
import six


def make_simple_mesh(n):
    entries = np.linspace(-1, 1, n)
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


# def rotate(nodes, dt):
#     c = np.cos(dt)
#     s = np.sin(dt)
#     M = np.array([
#         [c, s],
#         [-s, c],
#     ])
#     return nodes.dot(M)
def rotate_beta(x_vec):
    # x-values are in the first column, y-values in 2nd
    result = x_vec[:, ::-1].copy()
    result[:, 0] *= -1
    return result


def rk4_single(x_vec, ode, dt):
    k1 = ode(x_vec)
    k2 = ode(x_vec + 0.5 * dt * k1)
    k3 = ode(x_vec + 0.5 * dt * k2)
    k4 = ode(x_vec + dt * k3)
    return x_vec + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def rk4(x_vec, ode, T, n=10):
    dt = T / float(n)
    for _ in six.moves.xrange(n):
        x_vec = rk4_single(x_vec, ode, dt)
    return x_vec


def plot_mesh(triangles, *node_list):
    for nodes in node_list:
        plt.triplot(nodes[:, 0], nodes[:, 1], triangles)
        plt.plot(nodes[:, 0], nodes[:, 1], 'o')
    plt.axis('scaled')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.show()


def main():
    nodes, triangles = make_simple_mesh(4)
    new_nodes = rk4(nodes, rotate_beta, 0.4)
    plot_mesh(triangles, new_nodes, nodes)


main()
