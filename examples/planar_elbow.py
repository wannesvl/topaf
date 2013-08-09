import pathfollowing as pf
import numpy as np
from casadi import *

# System parameters
l1, l2 = 1., 1.
lc1, lc2 = 0.5, 0.5
m1, m2 = 1., 1.
I1, I2 = 0.001, 0.001
c1, c2 = 2., 1.
g = 9.81

# Define flat system
S = pf.FlatSystem(2, 2)
y = S.y

q1 = S.set_state(y[0, 0], 'q1')
q2 = S.set_state(y[1, 0], 'q2')
dq1 = S.set_state(y[0, 1], 'dq1')
dq2 = S.set_state(y[1, 1], 'dq2')
ddq1 = S.set_state(y[0, 2], 'ddq1')
ddq2 = S.set_state(y[1, 2], 'ddq2')
d11 = S.set_state(m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(q2)) + I1 + I2, 'd11')
d12 = S.set_state(m2*(lc2**2 + l1*lc2*cos(q2)) + I2, 'd12')
d22 = S.set_state(m2*lc2**2 + I2, 'd22')
c = S.set_state(-m2*l1*lc2*sin(q2), 'c')
g1 = S.set_state((m1 * lc1 + m2 * lc2) * g * cos(q1) + m2 * lc2 * g * cos(q1 + q2), 'g1')
g2 = S.set_state(m2 * lc2 * g * cos(q1 + q2), 'g2')
tau1 = S.set_state(d11 * ddq1 + d12 * ddq2 + c1 * dq1 + 2 * c * dq1 * dq2 + c * dq2 ** 2 + g1, 'tau1')
tau2 = S.set_state(d12 * ddq1 + d22 * ddq2 - c * dq1 ** 2 + g2 + c2 * dq2, 'tau2')

# Path following problem
P = pf.PathFollowing(S)
s = P.s[0]
s = s ** 2 * (s + 3 * (1 - s))
path = [0.5 * np.pi * s, -np.pi * s]
P.set_path(path)
P.set_constraint('tau1', -20, 20)
P.set_constraint('tau2', -10, 10)
P.set_options({'Nt': 199, 'N': 199, 'tol': 1e-8, 'reg': 0, 'bc': True})
P.solve()
