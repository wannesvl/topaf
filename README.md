TOPAF
=====
(Time) optimal path following for differentially flat systems

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
is `S.y[i, j]`. The instance method `set_state` allows to express the state
x of the system S as a function of the flat outputs:

    x = S.set_state(expr, name=None)

Optionally, the state can be given a name. The computation of time
derivatives is facilitated with the instance method `dt`:

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

For the flat system S an instance of a path following problem is
created via

    P = PathFollowing(S)

The symbolic path coordinate and its time derivatives are stored in the
instance attribute `s`. The reference path is set by the instance method

    set_path(expr)

where `expr` is a list in which each component of the flat output is
expressed as a function of `s[0]`.

Inequality constraints are set using the instance method

    set_constraint(expr, lb, ub)

where `lb` and `ub` are the lower and upper bounds. Furthermore, various
solver options are set through

    set_options(’option’: value)

Aside from all supported options of IPOPT in CasADi, the following
options are available

-   `N`: The number of discretization steps

-   `Nt`: The number of returned time points in the solution

-   `reg`: A regularization factor added to the goal function to avoid
    singular arcs in the solution

Finally, the instance method

    solve()

solves the problem. The solution is stored in the instance attribute
`sol`. The instance method

    plot()

plots all states as defined in the flat system and inequality
constraints.

Continuing the example from previous section, we define a path following
problem for the overhead crane tracking a circular trajectory with its
load. The velocities of the trolley and hoisting mechanisms are
constrained to [-5,5] and [-2.5,2.5] respectively.

    P = PathFollowing(S)
    path = [
        0.25 * sin(2 * pi * P.s[0]),
        0.25 * cos(2 * pi * P.s[0]) + 0.5
    ]
    P.set_path(path)
    P.set_constraint(du1, -5, 5)
    P.set_constraint(du2, -2.5, 2.5)
    P.set_options({'reg': 1e-10})
    P.solve()
    P.plot()

More Examples
-------------
Check out the examples directory!
