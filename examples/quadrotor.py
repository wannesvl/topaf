from pathfollowing import *
from casadi import *
from numpy import pi

G = 9.81
m = 1.35
Jx, Jy, Jz, I = 0.1325, 0.1325, 0.2651, 0.235

# Define flat system
Q = FlatSystem(4, 5)
y = Q.y
px = Q.set_state(y[0, 0], 'x')
py = Q.set_state(y[1, 0], 'y')
pz = Q.set_state(y[2, 0], 'z')
psi = Q.set_state(y[3, 0], 'psi')
u4 = Q.set_state(m * sqrt(y[0, 2] ** 2 + y[1, 2] ** 2 + (y[2, 2] - G) ** 2), 'u4')
theta = Q.set_state(arcsin(-m * (y[0, 2] * sin(y[3, 0]) - y[1, 2] * cos(y[3, 0])) / u4), 'theta')
phi = Q.set_state(arcsin(- m * (y[0, 2] * cos(y[3, 0]) + y[1, 2] * sin(y[3, 0])) / (u4 * cos(theta))), 'phi')
wx = Q.set_state(Q.dt(theta) - sin(phi) * y[3, 1], 'wx')
wy = Q.set_state(cos(theta) * Q.dt(phi) + sin(theta) * cos(phi) * y[3, 1], 'wy')
wz = Q.set_state(-sin(theta) * Q.dt(phi) + cos(theta) * cos(phi) * y[3, 1], 'wz')
u2 = Q.set_state((Jx * Q.dt(wx) + (Jz - Jy) * wy * wz) / I, 'u2')
u1 = Q.set_state((Jy * Q.dt(wy) + (Jx - Jz) * wx * wz) / I, 'u1')
u3 = Q.set_state(Jz * Q.dt(wz) + (Jy - Jx) * wx * wy, 'u3')

# Path tracking problem
P = PathFollowing(Q)
s = P.s[0]
alpha = 0.5
ps = 2 * pi * s ** 4 * (35 - 84 * s + 70 * s ** 2 - 20 * s ** 3)
path = [cos(ps), sin(ps), -(0.9 * (exp(ps / 2 / pi)-1) + 0.1 * sin(ps)) ** 2, ps]
P.set_path(path)
P.set_constraint('u1', -8, 8)
P.set_constraint('u2', -8, 8)
P.set_constraint('u3', -8, 8)
P.set_constraint('u4', 1, 32)
P.set_options({'Nt': 199, 'N': 199, 'reg': 1e-14})
P.solve()
P.plot([['u1', 'u2', 'u3', 'u4']])
