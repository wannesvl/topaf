TOPAF (Time) optimal path following for differentially flat systems
===================================================================

Installation
------------

The software is developed in the Python programming
language and uses [CasADi](https://github.com/casadi/casadi/wiki) as
a framework for symbolic computations. Furthermore CasADi provides an
interface to IPOPT, a software package for
large-scale nonlinear optimization. For installation instructions
regarding these software packages, the user is referred to the [CasADi
homepage](https://github.com/casadi/casadi/wiki). CasADi now offers
binaries, which simplify the installation procedure considerably.

Run

    >> sudo python setup.py install

to install our software.

Flat systems
------------

The class FlatSystem defines a flat system. A flat system S with m
flat outputs using derivatives up to order k is defined as

    S = FlatSystem(m, k)

Its symbolic flat output variables are stored in the instance attribute
y. For example, the j-th derivative of flat output i of system S
is `S.y[i, j]'. The instance method `set_state' allows to express the state
x of the system S as a function of the flat outputs:

    x = S.set~s~tate(expr, name=None)

Optionally, the state can be given a name. The computation of time
derivatives is facilitated with the instance method `dt':

    xdot = S.dt(x)

As an example, we model an overhead crane.

    from casadi import *
    from pathfollowing import *

    S = FlatSystem(2, 4)
    G = 9.81
    y1, dy1, ddy1 = S.y[0, 0], S.y[0, 1], S.y[0, 2]
    y2, dy2, ddy2 = S.y[1, 0], S.y[1, 1], S.y[1, 2]
    u1 = S.set_state(y1 + y2 * ddy1 / (G - ddy2), 'u1')
    u2 = S.set_state(y2 * sqrt(1 + (ddy1 / (G - ddy2)) ** 2), 'u2')
    theta = S.set_state(arctan(-ddy1 / (G - ddy2)), 'theta')
    du1 = S.set_state(S.dt(u1), 'du1')
    du2 = S.set_state(S.dt(u2), 'du2')

Path following
--------------

For the flat system $S$ an instance of a path following problem is
created via

P = PathFollowing(S)

The symbolic path coordinate and its time derivatives are stored in the
instance attribute s. The reference path is set by the instance method

set~p~ath(expr[, r2r=True])

where expr is a list in which each component of the flat output is
expressed as a function of s[0]. When r2r is True, the geometric path is
reparameterized as in [sec:singularities] such that a rest-to-rest
motion is imposed.

Inequality constraints are set using the instance method

set~c~onstraint(expr, lb, ub)

where lb and ub are the lower and upper bounds. Furthermore, various
solver options are set through

set~o~ptions(’option’: value)

Aside from all supported options of IPOPT in CasADi, the following
options are available

-   ’N’

    : The number of discretization steps

-   ’Nt’

    : The number of returned time points in the solution

-   ’reg’

    : A regularization factor added to the goal function to avoid
    singular arcs in the solution

Finally, the instance method

solve()

solves the problem. The solution is stored in the instance attribute
sol. The instance method

plot()

plots all states as defined in the flat system and inequality
constraints.

Continuing the example from previous section, we define a path following
problem for the overhead crane tracking a circular trajectory with its
load. The velocities of the trolley and hoisting mechanisms are
constrained to $[-5,5]
\, \si{\metre\per\second}$ and $[-2.5,2.5]\,
\si{\metre\per\second}$ respectively.

P = PathFollowing(S) path = [ 0.25 \* sin(2 \* pi \* P.s[0]), 0.25 \*
cos(2 \* pi \* P.s[0]) + 0.5 ] P.set~p~ath(path) P.set~c~onstraint(du1,
-5, 5) P.set~c~onstraint(du2, -2.5, 2.5) P.set~o~ptions(’reg’: 1e-10)
P.solve() P.plot()

Figure [fig:crane~v~el] shows the resulting velocities of the trolley
and hoisting mechanisms as a function of the path coordinate. At each
time instant at least one of the constraints is active, indicating a
time-optimal trajectory. Figure [fig:crane~o~pt] illustrates the
movement of the crane required to follow the circular trajectory in 10
equal time steps.

![image](controls_tikz) Time-optimal velocity signals as a function of
the path coordinate for an overhead crane following a circular
trajectory[fig:crane~v~el]

Time optimal path following of a circle for the overhead crane in 10
equal time steps [fig:crane~o~pt]

[o] ![image](movingcrane_tikz)

Path planning
-------------

A subclass of PathFollowing allows to define path planning problems. An
instance of a path planning problem for a differentially flat system $S$
is created via

Q = PathPlanning(S)

Similarly, the symbolic path coordinate and its derivatives are stored
in the instance attribute s.

The outer paths are set with the instance method

set~p~ath(expr[, r2r=True])

where expr is a list. Each element parameterizes one of the outer paths
by another list, in which each element contains a component of the flat
output as a function of s[0].

As before, constraints are set with the instance method

set~c~ontstraint(expr, lb, ub)

and solver options are set with the instance method

set~o~ptions(’option’: value)

An added option for path planning problems is ’Nc’, which controls the
number of spline coefficients used for the convex combination functions
$p_i(s)$ (cfr. Section [sec:path~p~lanning]).

Finally, calling the instance method

solve()

solves the path planning problem.

Let’s look at a path planning example for the two degree of freedom
robotic manipulator from Section [subsec:robotpathfollowing]. Similar to
previous example, we first define the robot as a flat system, for which
the code can be found in the online examples. Now, we want to move the
robot time optimally from an initial configuration $(0,0)^T$ to
$(\pi/2,0)^T$. To this end, we define the outer paths, before
reparameterization, as $$\begin{gathered}
y_1 = \left(\frac{\pi}{2} s, -32 (s - 0.5)^2 + 8 \right)^T  \\
y_2 = \left(\frac{\pi}{2} s, 32 (s - 0.5)^2 - 8 \right)^T .
\end{gathered}$$ Note that, for simplicity, we only allow freedom in the
movement of the second joint.

Suppose we have modeled the robot as the flat system S. Then the path
planning problem can be modeled as follows:

P = PathPlanning(S) s = P.s[0] p = s \*\* 2 \* (s + 3 \* (1 - s)) y1 =
[np.pi / 2 \* p, -32 \* (p - 0.5) \*\* 2 + 8] y2 = [np.pi / 2 \* p, 32
\* (p - 0.5) \*\* 2 - 8] P.set~p~ath([y1, y2]) P.set~c~onstraint(’tau1’,
-20, 20) P.set~c~onstraint(’tau2’, -10, 10) P.set~o~ptions(’N’: 99,
’Nc’: 10) P.solve()

Figure [fig:robot] shows the movement of the robot in 15 equal time
steps. As expected, in order to move as fast as possible, the inertia
for the first joint is lowered by drawing in the second joint.

Time optimal movement of a planar manipulator in 15 equal time steps
[fig:robot]

[o][10cm] ![image](planarelbow_ani_tikz)