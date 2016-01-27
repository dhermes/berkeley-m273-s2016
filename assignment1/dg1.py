"""Simple dg1 script from Persson."""


import matplotlib.pyplot as plt
import numpy as np


def dg1(n, porder, T, dt):
    h  = 1.0 / n
    nsteps = int(round(T / dt))

    if porder == 1:
        Mel = h/6 * np.array([[2, 1], [1, 2]])
        Kel = 0.5 * np.array([[-1, -1], [1, 1]])
        x = np.array([np.linspace(0, 1-h, n), np.linspace(h, 1, n)])
    elif porder == 2:
        Mel = h/30 * np.array([[4,2,-1], [2,16,2], [-1,2,4]])
        Kel = 1.0/6.0 * np.array([[-3,-4,1], [4,0,-4], [-1,4,3]])
        x = np.array([np.linspace(0, 1-h, n),
                      np.linspace(h/2, 1-h/2, n),
                      np.linspace(h, 1, n)])
    else:
        print "Error: porder not implemented"
        assert(0)

    u = np.exp(-(x - 0.5)**2 / 0.1**2)

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(-.1, 1.1))
    ax.plot(x, u, 'r', lw=2)
    plt.grid(True)
    plot_lines = ax.plot(x, u, 'b', lw=2)

    animation_args = (fig, animate)
    animation_kwargs = {
        'frames': nsteps + 1,
        'interval': 20,
        'blit': True,
        'fargs': [u, plot_lines, Kel, Mel, dt, n],
    }
    return animation_args, animation_kwargs


def animate(frame_number, u, plot_lines, Kel, Mel, dt, n):
    u_orig = u
    u0 = u
    for irk in range(4,0,-1):
        r = np.dot(Kel, u)
        r[-1] -= u[-1]
        r[0] += u[-1, np.hstack([n-1, np.arange(n-1)])]
        u = u0 + dt/irk * (np.linalg.solve(Mel, r))
    for index, line in enumerate(plot_lines):
        line.set_ydata(u[:, index])
    u_orig[:] = u  # Update the original data.
    return plot_lines
