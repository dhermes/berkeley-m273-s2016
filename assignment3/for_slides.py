import seaborn

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial


seaborn.set_palette('husl')

COLORS = seaborn.husl_palette(6)
YELLOW = COLORS[1]
GREEN = COLORS[2]
BLUE = COLORS[4]


class Animation1(object):

    def __init__(self, n, a, b, num_steps=10):
        self.n = n
        self.update_vec = np.array([a, b])
        min_x = min(-1, -1 + a)  # a * 1.0 (max time)
        max_x = max(1, 1 + a)
        width_x = max_x - min_x
        mid_x = 0.5 * (max_x + min_x)
        min_y = min(-1, -1 + b)
        max_y = max(1, 1 + b)
        width_y = max_y - min_y
        mid_y = 0.5 * (max_y + min_y)

        self.fig = plt.figure()
        self.ax = self.fig.gca()
        self.ax.axis('scaled')
        self.ax.set_xlim(mid_x - 0.55 * width_x,
                         mid_x + 0.55 * width_x)
        self.ax.set_ylim(mid_y - 0.55 * width_y,
                         mid_y + 0.55 * width_y)
        W, H = self.fig.get_size_inches()
        self.fig.set_size_inches(2 * W, H)

        self.curr_step = 0
        self.num_steps = num_steps
        self.dt = 1.0 / num_steps
        self.path_collection = None

    @property
    def plot_frames(self):
        return self.num_steps + 1

    def init_func(self):
        interval = np.linspace(-1, 1, self.n + 1)
        X, Y = np.meshgrid(interval, interval)
        X = X.flatten(order='F')
        Y = Y.flatten(order='F')
        self.path_collection = self.ax.scatter(X, Y, color=GREEN)
        # For ``init_func`` with ``blit`` turned on, the initial
        # frame should not have visible lines. See
        # http://stackoverflow.com/q/21439489/1068170 for more info.
        self.path_collection.set_visible(False)
        return self.path_collection,

    def update_plot(self, frame_number):
        if self.curr_step != frame_number:
            raise ValueError('Current step does not match '
                             'frame number', self.curr_step,
                             frame_number)

        self.curr_step += 1
        if frame_number == 0:
            # ``init_func`` creates lines that are not visible, to
            # address http://stackoverflow.com/q/21439489/1068170.
            # So in the initial frame, we make them visible.
            self.path_collection.set_visible(True)
            return self.path_collection,

        # Update the scatter offsets in-place.
        offsets = self.path_collection.get_offsets()
        offsets += self.dt * self.update_vec
        return self.path_collection,

    def inputs(self):
        args = (
            self.fig,
            self.update_plot,
        )
        kwargs = {
            'init_func': self.init_func,
            'frames': self.plot_frames,
            'interval': 20,
            'blit': True,
        }
        return args, kwargs


def example1():
    animate_obj = Animation1(4, 1.0, 1.5, num_steps=30)
    return animate_obj.inputs()


class Animation2(object):

    def __init__(self, n, num_steps=10):
        self.n = n

        self.fig = plt.figure()
        self.ax = self.fig.gca()
        self.ax.axis('scaled')
        # Max-radius = sqrt(1 + 1) = sqrt(2)
        self.ax.set_xlim(-1.55, 1.55)
        self.ax.set_ylim(-1.55, 1.55)
        W, H = self.fig.get_size_inches()
        self.fig.set_size_inches(2 * W, H)

        self.curr_step = 0
        self.num_steps = num_steps
        self.dt = 0.25 * np.pi / num_steps
        c_dt = np.cos(self.dt)
        s_dt = np.sin(self.dt)
        self.update_mat = np.array([
            [c_dt, -s_dt],
            [s_dt, c_dt],
        ])
        self.path_collection = None

    @property
    def plot_frames(self):
        return self.num_steps + 1

    def init_func(self):
        interval = np.linspace(-1, 1, self.n + 1)
        X, Y = np.meshgrid(interval, interval)
        X = X.flatten(order='F')
        Y = Y.flatten(order='F')
        self.path_collection = self.ax.scatter(X, Y, color=GREEN)
        # For ``init_func`` with ``blit`` turned on, the initial
        # frame should not have visible lines. See
        # http://stackoverflow.com/q/21439489/1068170 for more info.
        self.path_collection.set_visible(False)
        return self.path_collection,

    def update_plot(self, frame_number):
        if self.curr_step != frame_number:
            raise ValueError('Current step does not match '
                             'frame number', self.curr_step,
                             frame_number)

        self.curr_step += 1
        if frame_number == 0:
            # ``init_func`` creates lines that are not visible, to
            # address http://stackoverflow.com/q/21439489/1068170.
            # So in the initial frame, we make them visible.
            self.path_collection.set_visible(True)
            return self.path_collection,

        # Update the scatter offsets in-place.
        offsets = self.path_collection.get_offsets()
        new_offsets = offsets.dot(self.update_mat.T)
        self.path_collection.set_offsets(new_offsets)
        return self.path_collection,

    def inputs(self):
        args = (
            self.fig,
            self.update_plot,
        )
        kwargs = {
            'init_func': self.init_func,
            'frames': self.plot_frames,
            'interval': 20,
            'blit': True,
        }
        return args, kwargs


def example2():
    animate_obj = Animation2(4, num_steps=60)
    return animate_obj.inputs()


def example3():
    npz_file = np.load('example3.npz')
    N0 = npz_file['N0']
    T0 = npz_file['T0']
    N1 = npz_file['N1']
    T1 = npz_file['T1']
    plt.triplot(N0[:, 0], N0[:, 1], T0, color=GREEN)
    plt.triplot(N1[:, 0], N1[:, 1], T1, linestyle='dashed',
                color=YELLOW)
    plt.axis('scaled')

    fig = plt.gcf()
    W, H = fig.get_size_inches()
    fig.set_size_inches(1.5 * W, 1.5 * H)

    plt.show()


def example4():
    npz_file = np.load('example3.npz')
    N0 = npz_file['N0']
    T0 = npz_file['T0']
    N1 = npz_file['N1']
    T1 = npz_file['T1']
    plt.triplot(N0[:, 0], N0[:, 1], T0, color=GREEN)
    plt.triplot(N1[:, 0], N1[:, 1], T1, linestyle='dashed',
                color=YELLOW)

    for i, tri in enumerate(T1):
        centroid = np.mean(N1[tri, :], axis=0)
        plt.text(centroid[0], centroid[1], str(i))

    plt.axis('scaled')
    fig = plt.gcf()
    W, H = fig.get_size_inches()
    fig.set_size_inches(1.5 * W, 1.5 * H)

    plt.show()


def example5():
    npz_file = np.load('example5.npz')
    N0 = npz_file['N0']
    T0 = npz_file['T0']
    N1 = npz_file['N1']
    T1 = npz_file['T1']
    D = {
        1: npz_file['D1'],
        6: npz_file['D6'],
        7: npz_file['D7'],
        28: npz_file['D28'],
        29: npz_file['D29'],
        32: npz_file['D32'],
    }

    t1 = N0[T0[1, :], :]
    plt.plot(t1[[0, 1, 2, 0], 0], t1[[0, 1, 2, 0], 1],
             color=GREEN, zorder=0)

    for j in (1, 6, 7, 28, 29, 32):
        tj = N1[T1[j, :], :]
        line, = plt.plot(tj[[0, 1, 2, 0], 0], tj[[0, 1, 2, 0], 1],
                         color=YELLOW, linestyle='dashed', zorder=0)
        centroid = np.mean(tj, axis=0)
        plt.text(centroid[0], centroid[1], str(j))

    D_vals = np.vstack(D.values())
    plt.scatter(D_vals[:, 0], D_vals[:, 1],
                color='black', s=20, zorder=1)

    plt.axis('scaled')
    fig = plt.gcf()
    W, H = fig.get_size_inches()
    fig.set_size_inches(1.5 * W, 1.5 * H)

    plt.show()


def example6():
    npz_file = np.load('example5.npz')
    N0 = npz_file['N0']
    T0 = npz_file['T0']
    N1 = npz_file['N1']
    T1 = npz_file['T1']
    D28 = npz_file['D28']

    t1 = N0[T0[1, :], :]
    plt.plot(t1[[0, 1, 2, 0], 0], t1[[0, 1, 2, 0], 1],
             color=GREEN, zorder=0)
    t28 = N1[T1[28, :], :]
    line, = plt.plot(t28[[0, 1, 2, 0], 0], t28[[0, 1, 2, 0], 1],
                     color=YELLOW, linestyle='dashed', zorder=0)
    centroid = np.mean(t28, axis=0)
    plt.text(centroid[0], centroid[1], '28')

    delaunay_tri = scipy.spatial.Delaunay(D28)
    tri_local = delaunay_tri.simplices
    plt.triplot(D28[:, 0], D28[:, 1], tri_local, color=BLUE)

    plt.axis('scaled')
    fig = plt.gcf()
    W, H = fig.get_size_inches()
    fig.set_size_inches(1.5 * W, 1.5 * H)

    plt.show()


def example6a():
    npz_file = np.load('example5.npz')
    N0 = npz_file['N0']
    T0 = npz_file['T0']
    N1 = npz_file['N1']
    T1 = npz_file['T1']

    t1 = N0[T0[1, :], :]
    plt.plot(t1[[0, 1, 2, 0], 0], t1[[0, 1, 2, 0], 1],
             color=GREEN, zorder=0)
    for j in (1, 28):
        tj = N1[T1[j, :], :]
        line, = plt.plot(tj[[0, 1, 2, 0], 0], tj[[0, 1, 2, 0], 1],
                         color=YELLOW, linestyle='dashed', zorder=0)
        centroid = np.mean(tj, axis=0)
        plt.text(centroid[0], centroid[1], str(j))

    # NOTE: This is t28[2, :] and t1[2, :]
    plt.scatter([-0.96172635972908305], [-0.43150147173950337],
                color='black', zorder=1)

    plt.axis('scaled')
    fig = plt.gcf()
    W, H = fig.get_size_inches()
    fig.set_size_inches(1.5 * W, 1.5 * H)

    plt.show()


def example7():
    npz_file = np.load('example7.npz')
    N = npz_file['N']
    T = npz_file['T']

    plt.triplot(N[:, 0], N[:, 1], T, color=GREEN)
    plt.plot(N[:, 0], N[:, 1], 'o', color=YELLOW)
    plt.title(r'$\Delta t = 0.1$', fontsize=20)

    plt.axis('scaled')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

    fig = plt.gcf()
    W, H = fig.get_size_inches()
    fig.set_size_inches(1.5 * W, 1.5 * H)

    plt.show()


def example8():
    npz_file = np.load('example8.npz')
    N = npz_file['N']
    T = npz_file['T']

    plt.triplot(N[:, 0], N[:, 1], T, color=GREEN)
    plt.plot(N[:, 0], N[:, 1], 'o', color=YELLOW)
    plt.title(r'$\Delta t = 0.2$', fontsize=20)

    plt.axis('scaled')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

    fig = plt.gcf()
    W, H = fig.get_size_inches()
    fig.set_size_inches(1.5 * W, 1.5 * H)

    plt.show()


def example9():
    npz_file = np.load('example9.npz')
    N = npz_file['N']
    T = npz_file['T']
    U = npz_file['U']

    plt.tricontourf(N[:, 0], N[:, 1], T, U, 20, cmap='viridis')

    plt.axis('scaled')
    plt.colorbar()

    fig = plt.gcf()
    W, H = fig.get_size_inches()
    fig.set_size_inches(1.5 * W, 1.5 * H)

    plt.show()
