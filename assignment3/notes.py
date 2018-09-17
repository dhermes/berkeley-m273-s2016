from __future__ import print_function

import six
import sympy


x, y = sympy.symbols('x, y')
phi = {
    0: 1 - x - y,
    1: x,
    2: y,
}
M = sympy.zeros(3)
for i in six.moves.xrange(3):
    for j in six.moves.xrange(3):
        p = phi[i] * phi[j]
        I = sympy.integrate(sympy.integrate(p, (x, 0, 1 - y)), (y, 0, 1))
        M[i, j] = I
print(repr(24 * M))

K = sympy.zeros(3)
for i in six.moves.xrange(3):
    grad = x * phi[i].diff(y) - y * phi[i].diff(x)
    for j in six.moves.xrange(3):
        p = phi[j] * grad
        I = sympy.integrate(sympy.integrate(p, (x, 0, 1 - y)), (y, 0, 1))
        K[i, j] = I
print(repr(24 * K))

s = sympy.Symbol('s')
G0 = sympy.zeros(3)
G1 = sympy.zeros(3)
G2 = sympy.zeros(3)
for i in six.moves.xrange(3):
    for j in six.moves.xrange(3):
        p = phi[i] * phi[j]
        g0 = (-x * p).subs({x: s, y: 0})
        g1 = ((x - y) * p).subs({x: 1 - s, y: s})
        g2 = (y * p).subs({x: 0, y: 1 - s})
        G0[i, j] = sympy.integrate(g0, (s, 0, 1))
        G1[i, j] = sympy.integrate(g1, (s, 0, 1))
        G2[i, j] = sympy.integrate(g2, (s, 0, 1))
print(repr(12 * G0))
print(repr(6 * G1))
print(repr(12 * G2))
G = G0 + G1 + G2
print(repr(12 * G))
