import pathfollowing as pf
import numpy as np
from casadi import *

# System parameters
# =================
l1  = 1
l2  = 1
lc1 = 0.5
lc2 = 0.5
m1  = 1
m2  = 1
g   = 9.81
I1  = 0.001
I2  = 0.001
c1  = 2
c2  = 1

# Define flat system
S = pf.FlatSystem(2,2)
y = S.y

S.set_state(y[0,0],'q1')
S.set_state(y[1,0],'q2')
S.set_state(y[0,1],'dq1')
S.set_state(y[1,1],'dq2')
S.set_state(y[0,2],'ddq1')
S.set_state(y[1,2],'ddq2')
S.set_state(m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*cos(y[1,0])) + I1 + I2,
           'd11')
S.set_state(m2*(lc2**2 + l1*lc2*cos(y[1,0])) + I2, 'd12')
S.set_state(m2*lc2**2 + I2, 'd22')
S.set_state(-m2*l1*lc2*sin(y[1,0]), 'c')
S.set_state((m1*lc1+m2*lc2)*g*cos(y[0,0]) + m2*lc2*g*cos(y[0,0]+y[1,0]), 'g1')
S.set_state(m2*lc2*g*cos(y[0,0]+y[1,0]), 'g2')
S.set_state(S.x['d11']*y[0,2] + S.x['d12']*y[1,2] + c1 * y[0, 1] +
        2*S.x['c']*y[0,1]*y[1,1]+S.x['c']*y[1,1]**2 + S.x['g1'], 'tau1')
S.set_state(S.x['d12']*y[0,2] + S.x['d22']*y[1,2] - S.x['c']*y[0,1]**2 +
           S.x['g2'] + c2 * y[1, 1], 'tau2')

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
