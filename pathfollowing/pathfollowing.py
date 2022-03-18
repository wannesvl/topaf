# TOPAF -- Time optimal path following for differentially flat systems
# Copyright (C) 2013 Wannes Van Loock, KU Leuven. All rights reserved.
#
# TOPAF is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA

"""
Time optimal path following
===========================

This module exports two classes:
    * FlatSystem: Defines a differentially flat system
    * PathFollowing: Defines a path following problem for a FlatSystem object
and exports the function:
    * evalf: Evaluate casadi function at a point

Run

    > python setup.py install

to install the software systemwide

Dependencies:
    * CasADi and Ipopt are required. Precompiled binaries containing CasADi and
      Ipopt are available on the CasADi homepage: http://casadi.org/

.. moduleauthor:: Wannes Van Loock <wannes.vanloock@gmail.com>
"""

import casadi as cas
import numpy as np
import warnings
#from scipy.misc import comb
from scipy.special import factorial, comb

fact = factorial(range(20))


class FlatSystem(object):
    """Define a flat system (with CasADi variables)

    Args:
        ny (int): number of flat outputs
        order (int): order of highest derivative needed to define states and inputs

    Instance variables:
        * order (int): order of the flat system
        * ny (int): number of flat outputs
        * y (SX): the flat output and its time derivatives
        * x (dict): the states as a function of the flat outputs and its derivatives

    Example:
        >>> S = FlatSystem(2, 4)
        >>> print S
        Flat system of order 4 with 2 flat outputs
    """
    def __init__(self, ny, order):
        self.order = order
        self.ny = ny
        self.y = cas.SX.sym("y", ny, self.order + 1)
        self.x = dict()
        self._nx = 0
        self._dy = cas.horzcat(self.y[:, 1:], cas.SX.zeros(ny, 1))

    def __str__(self):
        return "Flat system of order %s with %s flat outputs" % (self.order, self.ny)

    __repr__ = __str__

    def set_state(self, expr, name=None):
        """Update the dictionary of states with expr

        Args:
            expr (SX): casadi expression for the state
            name (str): Dictionary key to access state. (Default 'xn')

        Note: States are stored in the instance variable ``x`` as a dictionary
        with keys the names of the states

        Example:
            >>> S = FlatSystem(2, 4)
            >>> S.set_state(S.y[0, 0], name='state1')
            >>> print S.x['state1']
            y0
        """
        if name is None:
            name = ''.join(['x', str(self._nx)])
        self.x.update({name: expr})
        self._nx += 1

        return expr

    def dt(self, expr):
        """Return time derivative of expr

        The time derivative is computed using the chainrule:
        df/dt = df/dy * dy/dt

        Args:
            expr (SX): casadi expression that is differentiated wrt time

        Returns:
            SX. The time derivative of expr

        Example:
            >>> S = FlatSystem(2, 4)
            >>> dy00 = S.dt(S.y[0, 0])
        """
        return cas.jtimes(expr, self.y, self._dy)


class PathFollowing(object):
    """Define and solve path following problems

    Args:
        sys (FlatSystem): The associated flat system

    Instance variables
        * sys (FlatSystem): associated flat system
        * s (SX): The path coordinate and its time derivatives as a
          casadi symbolic object
        * path (SX): The path to be followed and its derivatives. Set by
          function setPath
        * constraints (list): List of tuples of constraints.
          (constrainfunction, lower bound, upper bound)
        * options (dict): Dictionary of options that is passed to the solver.
          Aside from the Ipopt options, following options are allowed:
            * N (int): Number of descritization steps in optimization problem
              (default=199)
            * Nt (int): Number of equidistant time points in the solution
              (default=499)
            * reg (double): Regularization constant added to the goal
              function to avoid ringing in singular arcs (default 1e-10)
        * sol (dict): Dictionary containing solution with keys:
            * t (numpy.array): The time vector grid
            * s (numpy.array): The path coordinate as a function of time
            * states (dict): The states as a function of time. The keys
              are the same as those of sys.x
            * diagnostics: not implemented yet
        * prob (dict): Problem definition

    Example:
        >>> S = FlatSystem(2, 4)
        >>> P = PathFollowing(S)

    .. todo:: implement setObjective function
    .. todo:: evaluate path and derivatives beforehand
    """
    def __init__(self, sys):
        self.sys = sys
        self.s = cas.SX.sym("s", self.sys.order + 1)
        self.path = cas.SX.nan(self.sys.y.shape[0], self.sys.order + 1)
        self.constraints = []
        self.objective = {'Lagrange': [], 'Mayer': []}
        self.options = {
            'N': 199, 'Ts': 0.01, 'solver': 'ipopt', 'solver_options': {'ipopt.tol': 1e-6,
            'ipopt.max_iter': 100}, 'plot': True, 'reg': 1e-20, 'bc': False
            }
        self.sol = {
            's': [], 't': [], 'states': [], 'inputs': [], 'diagnostics': 0
            }
        self.prob = {
            'var': [], 'con': [], 'obj': [], 'solver': [], 'x_init': None,
            's': [], 'path': []
            }

    def __str__(self):
        return "Path following problem for \"%s\"" % (self.sys)

    __repr__ = __str__

    # ========================================================================
    # Problem definition
    # ========================================================================
    def set_path(self, path, r2r=False):
        """Define an analytic expression of the geometric path.

        Note: The path must be defined as a function of self.s[0].

        Args:
            path (list of SX): An expression of the geometric path as
                a function of self.s[0]. Its dimension must equal self.sys.ny
            r2r (boolean): Reparameterize path such that a rest to rest
                transition is performed

        Example:
            >>> S = FlatSystem(2, 4)
            >>> P = PathFollowing(S)
            >>> P.set_path([P.s[0], P.s[0]])
        """
        if isinstance(path, list):
            path = cas.vcat(path)
        if r2r:
            path = cas.substitute(path, self.s[0], self._r2r())
        self.path[:, 0] = path
        dot_s = cas.vcat([self.s[1:], 0])
        for i in range(1, self.sys.order + 1):
            self.path[:, i] = cas.jtimes(self.path[:, i - 1], self.s, dot_s)

    def _r2r(self):
        """Reparameterization such that the path defines a rest to rest
        transition"""
        degree = 2 * self.sys.order - 1
        bernstein_coeffs = np.hstack((
            [round(comb(degree, i)) for i in range((degree + 1) // 2)],
            [0] * ((degree - 1) // 2),
            0))
        return sum([bernstein_coeffs[i] * self.s[0] ** (degree - i) *
                    (1 - self.s[0]) ** i for i in range(degree + 1)])

    def set_options(self, options):
        """Update dictionary of options

        Args:
            options (dict): options dictionary

        Example:
            >>> S = FlatSystem(2, 4)
            >>> P = PathFollowing(S)
            >>> P.set_options({'N': 299, 'reg': 1e-15, 'tol': 1e-6})
        """
        self.options.update(options)

    def set_constraint(self, expr, lb, ub, pos=None):
        """Append constraint constraint to list of constraints

        If expr is the name of a state of self.sys append it to the list of
        constraints, otherwise expr must be expressed as a function of
        self.sys.y

        Args:
            expr (SX): Name of the state or expression as a
                function of self.sys.y

            lb (double | SX): Lower bound for expr,
                either as double of as a function of self.s[0]

            ub (double | SX): Lower bound for expr,
                either as double of as a function of self.s[0]

            pos (list of ints): s coordinates where constraint 
                must be enforced

        Example:
            >>> S = FlatSystem(2, 4)
            >>> S.set_state(S.y[0,0], 'y0')
            >>> P = PathFollowing(S)
            >>> P.set_constraint('y0', -1, 1)
        """
        self.constraints.append((expr, lb, ub, pos))

    def empty_constraints(self):
        """Empty the list of constraints"""
        self.constraints = []

    def set_objective(self, expr, m_or_l):
        """Add terms to the objective function

        Note that Mayer terms are not yet supported!

        Args:
            expr (SX): The casadi expression for the objective terms
            m_or_l (str): 'Mayer' or 'Lagrange'
        """
        self.objective[m_or_l].append(expr)

    def set_grid(self, grid=None):
        """Set discretization grid for the optimal control problem.

        Define the discretization grid for the optimal control problem. The
        grid is a monotonically increasing list from 0 to 1. When the supplied
        grid is not consistent with self.options['N'] the value of 'N' is
        updated If grid is None the grid is automatically defined from
        self.options['N']

        Args:
            grid (list | numpy.array): a monotonically increasing list from 0
                to 1 (default=None)

        Example:
            >>> S = FlatSystem(2, 4)
            >>> P = PathFollowing(S)
            >>> P.set_grid([0, 0.25, 0.5, 0.75, 1])
            >>> print P.prob['grid']
            [0, 0.25, 0.5, 0.75, 1]
            >>> P.set_options({'N': 11})
            >>> P.set_grid()
            >>> print P.prob['grid']
            [ 0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1. ]
        """
        if grid is None:
            self.prob['s'] = np.linspace(0, 1, self.options['N'] + 1)
        else:
            if grid[-1] != 1:
                grid = np.append(grid, [1])
            self.prob['s'] = np.array(grid)
            self.options['N'] = grid.size - 1

    # ========================================================================
    # Problem solving
    # ========================================================================
    def check_ss_feasibility(self, tol=1e-6):
        """Returns whether path is steady state feasible.

        Args:
            tol (double): tolerance for feasibility check (default = 1e-6)

        Raises:
            warning when bounds may be too strict
        """
        for i, f in enumerate(self.constraints):
            F = cas.substitute(f[0], self.sys.y, self.path)
            F1 = cas.Function('F1',[self.s], [F - f[1]])
            F2 = cas.Function('F2',[self.s], [-F + f[2]])
            import ipdb
            ipdb.set_trace()
            c1 = np.array([F1(np.hstack([s, [1e-50] * (self.sys.order)])).T
                           for s in self.prob['s']])
            c2 = np.array([F2(np.hstack([s, [1e-50] * (self.sys.order)])).T
                           for s in self.prob['s']])
            if np.any(c1 + tol < 0):
                warnings.warn("lower bound of constraint %d may be to strict" % i)
                return False
            elif np.any(c2 + tol < 0):
                warnings.warn("upper bound of constraint %d may be to strict" % i)
                return False
        return True

    def solve(self):
        """Solve the optimal control problem

        solve() first check for steady state feasibility and defines the
        optimal control problem. After solving, the instance variable sol can
        be used to examine the solution.

        TODO: Add support for other solvers
        TODO: version bump
        """
        if len(self.prob['s']) == 0:
            self.set_grid()
        # Check feasibility
        # self.check_ss_feasibility()
        # Construct optimization problem
        self.prob['solver'] = None
        N = self.options['N']
        self.prob['vars'] = cas.SX.sym("b", N + 1, self.sys.order)
        V = cas.vec(self.prob['vars'])
        self._make_objective()
        self._make_constraints()

        nlp = {"x": V, "f": self.prob['obj'], "g": self.prob['con'][0]}
        solver = cas.nlpsol("solver",self.options.get('solver'),nlp,self.options.get('solver_options'))
        # Setting constraints
        solver_args = {}
        solver_args["lbg"] = cas.vcat(self.prob['con'][1])
        solver_args["ubg"] = cas.vcat(self.prob['con'][2])
        solver_args["ubx"] = [np.inf] * self.sys.order * (N + 1)
        solver_args["lbx"] = cas.vcat((
                [0] * (N + 1),
                (self.sys.order - 1) * (N + 1) * [-np.inf]))

        sol = solver(**solver_args)
        self.prob['solver'] = solver
        self.prob['sol'] = sol
        self._get_solution()

    # ========================================================================
    # Constraint handling
    # ========================================================================
    def _make_constraints(self):
        """
        Parse the constraints and put them in the correct format
        """
        N = self.options['N']
        con = self._ode(self.prob['vars'])
        lb = con.numel() * [0]
        ub = con.numel() * [0]
        S = self.prob['s']
        b = self.prob['vars']
        path, bs = self._make_path()[0:2]
        sbs = cas.vcat([self.s[0], bs])
        Sb = cas.hcat([S, b]).T
        for f in self.constraints:
            F = cas.substitute(f[0], self.sys.y, path)
            if f[3] is None:
                F = cas.vcat([cas.substitute(F, sbs, Sb[:, j])
                                 for j in range(N + 1)])
                Flb = f[1](S) if hasattr(f[1], '__call__') else [f[1]] * len(S)
                Fub = f[2](S) if hasattr(f[2], '__call__') else [f[2]] * len(S)
                con = cas.vertcat(con,F)
                lb.extend(Flb)
                ub.extend(Fub)
            else:
                F = cas.vcat([cas.substitute(F, sbs, Sb[:, j])
                                 for j in f[3]])
                con = cas.vertcat(con,F)
                lb.extend([f[1]])
                ub.extend([f[2]])
        if self.options['bc']:
            con.append(b[0, 0])
            lb.append(0)
            ub.append(1e-50)
            con.append(b[-1, 0])
            lb.append(0)
            ub.append(1e-50)
        self.prob['con'] = [con, lb, ub]

    def _ode(self, v):
        """Define ode for linear brunovsky system

        Args:
            v (SX): The variables for the ode

        Returns:
            SX. The ode for the optimal control problem
        """
        n = v.size2()
        N = self.options['N']
        # delta = 1.0/(N+1)
        delta = np.diff(self.prob['s'])
        o = []
        for i in range(1, n):
            f = v[:N, n - 1] / fact[i]
            for j in range(1, i + 1):
                f = f * delta + v[:N, n - j - 1] / fact[i - j]
            f = f - v[1:, n - 1 - i]
            o = cas.vcat((o, f))
        return o

    def _make_path(self):
        """Rewrite the path as a function of the optimization variables.

        Substitutes the time derivatives of s in the expression of the path by
        expressions that are function of b and its path derivatives by
        repeatedly applying the chainrule

        Returns:
            * SX. The substituted path
            * SX. b and the path derivatives
            * SX. The derivatives of s as a function of b
        """
        b = cas.SX.sym("b", self.sys.order)
        db = cas.vcat((b[1:], 0))
        Ds = cas.SX.nan(self.sys.order)  # Time derivatives of s
        Ds[0] = cas.sqrt(b[0])
        Ds[1] = b[1] / 2
        # Apply chainrule for finding higher order derivatives
        for i in range(1, self.sys.order - 1):
            Ds[i + 1] = (cas.jtimes(Ds[i], b, db) * self.s[1] +
                       cas.jacobian(Ds[i], self.s[1]) * Ds[1])
        Ds = cas.substitute(Ds, self.s[1], cas.sqrt(b[0]))
        return cas.substitute(self.path, self.s[1:], Ds), b, Ds

    # ========================================================================
    # Objective function
    # ========================================================================
    def _make_objective(self):
        """Construct objective function from the problem definition

        Make time optimal objective function and add a regularization to
        ensure a unique solution. When additional objective terms are defined
        these are added to the objective as well.

        TODO: Improve accuracy of integration
        """
        order = self.sys.order
        N = self.options['N']
        b = self.prob['vars']
        # ds = 1.0/(N+1)
        obj = 2 * cas.sum1(np.diff(self.prob['s']) / (cas.sqrt(b[:N, 0]) + cas.sqrt(b[1:, 0])))
        # reg = sum(cas.sqrt((b[1:, order - 1] - b[:N, order - 1]) ** 2))
        reg = cas.sum1((b[2:, order - 1] - 2 * b[1:N, order - 1] +
                    b[:(N - 1), order - 1]) ** 2)
        for f in self.objective['Lagrange']:
            path, bs = self._make_path()[0:2]
            # S = np.arange(0, 1, 1.0/(N+1))
            S = self.prob['s']
            b = self.prob['vars']
            L = cas.substitute(f, self.sys.y, path)
            L = 2 * sum(cas.vcat([
                            cas.substitute(
                                L,
                                cas.vcat([self.s[0], bs]),
                                cas.vcat([S[j], b[j, :].T])
                            ) for j in range(1, N + 1)
                        ])
                    * np.diff(self.prob['s']) / (cas.sqrt(b[:N, 0]) + cas.sqrt(b[1:, 0]))
                )
            # L = sum(cas.vcat([cas.substitute(L, cas.vcat([self.s[0],
            #     [bs[i] for i in range(0, bs.numel())]]),
            #     cas.vcat([S[j], [b[j, i] for i in range(0, self.sys.order)]]))
            #     for j in range(0, N + 1)]))
            obj = obj + L
        self.prob['obj'] = obj + self.options['reg'] * reg

    # ========================================================================
    # Problem solution
    # ========================================================================
    def _get_solution(self):
        """Get the solution from the solver output

        Fills the dictionary self.sol with the information:
            * 's': The optimal s as a function of time
            * 't': The time vector
            * 'states': Numerical values of the states defined in self.sys

        TODO: perform accurate integration to determine time
        TODO: Do exact interpolation
        """
        solver = self.prob['solver']
        sol = self.prob["sol"]
        N = self.options['N']
        x_opt = np.array(sol["x"]).ravel()
        b_opt = np.reshape(x_opt, (N + 1, -1), order='F')
        self.sol['b'] = b_opt

        # Determine time on a sufficiently fine spatial grid
        s0 = np.linspace(0, 1, 1001)
        delta = s0[1] - s0[0]

        pieces = [lambda s, b=b_opt, ss=ss, j=j:
                  sum([bb * (s - ss) ** i / fact[i]
                       for i, bb in enumerate(b[j])])
                  for j, ss in enumerate(self.prob['s'][:-1])]
        conds = lambda s0: [np.logical_and(self.prob['s'][i] <= s0,
                                           s0 <= self.prob['s'][i+1])
                            for i in range(N)]
        b0_opt = np.piecewise(s0, conds(s0), pieces)
        b0_opt[b0_opt < 0] = 0
        time = np.cumsum(np.hstack([0, 2 * delta / (np.sqrt(b0_opt[:-1]) +
                                    np.sqrt(b0_opt[1:]))]))

        # Resample to constant time-grid
        t = np.arange(time[0], time[-1], self.options['Ts'])
        st = np.interp(t, time, s0)
        # Evaluate solution on equidistant time grid
        b_opt = np.c_[[np.piecewise(st, conds(st), pieces, b=b_opt[:, i:])
                       for i in range(self.sys.order)]].T
        st = np.matrix(st)

        # Determine s and derivatives from b_opt
        b, Ds = self._make_path()[1:]
        Ds_f = cas.Function('Ds',[b], [Ds])  # derivatives of s wrt b
        s_opt = np.hstack((st.T, np.array([np.array(Ds_f(bb)).ravel()
                                          for bb in b_opt])))
        self.sol['s'] = np.asarray(s_opt)
        self.sol['t'] = t
        # Evaluate the states
        f = cas.Function('f',[self.s], [cas.substitute(cas.vcat(self.sys.x.values()),
                                           self.sys.y, self.path)])
        f_val = np.array([np.array(f(s.T)).ravel() for s in s_opt])
        self.sol['states'] = dict([(k, f_val[:, i]) for i, k in
                          enumerate(self.sys.x.keys())])

    def plot(self, p=[['states'], ['constraints']], tikz=False, show=True):
        """Plot states and/or constraints.

        Args:
            p (list of lists): Each inner list defines a plot window. The
              inner lists either contain the strings 'states' and/or
              'constraints', the name of a state defined in ``self.sys`` or a
              CasADi expression in terms of the flat outputs. When giving a
              CasADi expression you can optionally supply a y-label by passing
              it as a tuple.

            tikz (boolean): When true the plot is saved to a .tikz file
              which you can include in you latex documents to produce
              beautiful plots (default False)

            show (boolean): When True the plot is shown by invoking show()
              from matplotlib (default True)
        """
        import matplotlib.pyplot as plt
        for plots in p:
            fig = plt.figure()
            j = 1
            m = (len(plots) + ('states' in plots) * (self.sys._nx - 1) +
                ('constraints' in plots) * (len(self.constraints) - 1))
            for l in plots:
                if l in self.sys.x.keys():
                    ax = fig.add_subplot(m / int(np.sqrt(m)) + np.mod(m, 2),
                                         int(np.sqrt(m)), j)
                    ax.plot(self.sol['t'], self.sol['states'][l])
                    plt.ylabel(l)
                    plt.xlabel('time')
                    j += 1
                elif (l == 'states'):
                    for i, k in enumerate(self.sol[l].keys()):
                        ax = fig.add_subplot(m / int(np.sqrt(m)) + np.mod(m, 2),
                                int(np.sqrt(m)), j)
                        ax.plot(self.sol['t'], self.sol[l][k])
                        plt.ylabel(k)
                        plt.xlabel('time')
                        j += 1
                elif l == 'constraints':
                    for i, f in enumerate(self.constraints):
                        F = cas.substitute(f[0], self.sys.y, self.path)
                        F = cas.Function('F',[self.s], [F])
                        Flb = cas.Function('Flb',[self.s], [f[1]])
                        Fub = cas.Function('Fub',[self.s], [f[2]])
                        c = np.array([np.array(F(s.T)).ravel()
                                               for s in self.sol['s']])
                        cub = np.array([np.array(Fub(s.T)).ravel()
                                               for s in self.sol['s']])
                        clb = np.array([np.array(Flb(s.T)).ravel()
                                               for s in self.sol['s']])
                        ax = fig.add_subplot(m / int(np.sqrt(m)) + np.mod(m, 2),
                                             int(np.sqrt(m)), j)
                        ax.plot(self.sol['t'], c)
                        ax.plot(self.sol['t'], clb, color='black')
                        ax.plot(self.sol['t'], cub, color='black')
                        j += 1
                elif isinstance(l, tuple):
                    if isinstance(l[1], cas.SX):
                        F = cas.substitute(l[1], self.sys.y, self.path)
                        F = cas.Function('F',[self.s], [F])
                        c = np.array([np.array(F(s.T)).ravel()
                                               for s in self.sol['s']])
                        ax = fig.add_subplot(m / int(np.sqrt(m)) + np.mod(m, 2),
                                             int(np.sqrt(m)), j)
                        ax.plot(self.sol['t'], c)
                    else:
                        ax = fig.add_subplot(m / int(np.sqrt(m)) + np.mod(m, 2),
                                    int(np.sqrt(m)), j)
                        ax.plot(self.sol['t'], l[1])
                        plt.ylabel(l[0])
                        plt.xlabel('time')
                    j += 1
                else:
                    if isinstance(l, cas.SX):
                        F = cas.substitute(l, self.sys.y, self.path)
                        F = cas.Function('F',[self.s], [F])
                        c = np.array([np.array(F(s.T)).ravel()
                                               for s in self.sol['s']])
                        ax = fig.add_subplot(m / int(np.sqrt(m)) + np.mod(m, 2),
                                             int(np.sqrt(m)), j)
                        ax.plot(self.sol['t'], c)
                    else:
                        ax = fig.add_subplot(m / int(np.sqrt(m)) + np.mod(m, 2),
                                    int(np.sqrt(m)), j)
                        ax.plot(self.sol['t'], l)
                        plt.xlabel('time')
                    j += 1
        if tikz and len(p) == 1:
            from matplotlib2tikz import save as tsave
            tsave('_'.join(p[0]) + '.tikz')
        if show:
            plt.show()

    def save(self, filename):
        """
        Save time vector states and inputs to file filename
        """
        from itertools import chain as ch
        f = open(filename, 'w')
        f.write('Time, ' + ', '.join(ch(self.sol['states'].keys())) + ', ' +
                ', '.join(ch(self.sol['inputs'].keys())) + '\n') 
        np.savetxt(f, np.vstack((self.sol['t'],
                                 self.sol['states'].values(),
                                 self.sol['inputs'].values())).T,
                      delimiter=', ')
        f.close()
