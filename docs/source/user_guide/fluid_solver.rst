Fluid Flow Solvers
===================

For convenience, we provide a simple implementation of fluid flow equations in
:class:`echemfem.FlowSolver`, which can be used to obtain a velocity field for
the :class:`echemfem.EchemSolver`.

Navier-Stokes Solver
--------------------

The :class:`echemfem.NavierStokesFlowSolver` class contains an implementation
for the steady-state incompressible Navier-Stokes equations. Both
nondimensional and dimensional equations are available.

We are solving for velocity :math:`\mathbf u` and pressure :math:`p`. The
dimensional version of the momentum equation is

.. math::

   -\nu \nabla^2 \mathbf u + \mathbf u \cdot \nabla \mathbf u + \frac{1}{\rho} \nabla p = 0,

where :math:`\nu` is the kinematic viscosity and :math:`\rho` is the density.
It is also common to use the dynamic viscosity :math:`\mu = \nu \rho`

The nondimensional version is

.. math::

   - \frac{1}{\mathrm{Re}}\nabla^2 \mathbf u + \mathbf u \cdot \nabla \mathbf u + \nabla p = 0,

where all quantities have been nondimensionalized, and :math:`\mathrm{Re}=
\frac{\bar {\mathbf u} L}{\nu}` is the Reynolds number, where :math:`L` is the
characteristic length.

In both cases, we also have the incompressibility condition

.. math::

   \nabla \cdot \mathbf{u} = 0.

Physical parameters are passed as a
:py:class:`dict` in the ``fluid_params`` argument.
Specifically, for the dimensional version, the user must pass the following keys:

* ``"density"``: :math:`\rho`

* ``"dynamic viscosity"`` or ``"kinematic viscosity"``: :math:`\mu` and :math:`\nu`, respectively


To use the nondimensional version, the user must pass the following key:

* ``"Reynolds number"``: :math:`\mathrm{Re}`

Navier-Stokes-Brinkman Solver
-----------------------------

The :class:`echemfem.NavierStokesBrinkmanFlowSolver` class contains an
implementation for the steady-state incompressible Navier-Stokes-Brinkman
equations. Both nondimensional and dimensional equations are available.

The dimensional version of the momentum equation is

.. math::

   -\nabla\cdot\left(\nu_\mathrm{eff} \nabla\mathbf u \right)+ \mathbf u \cdot \nabla \mathbf u + \frac{1}{\rho} \nabla p + \nu K^{-1} \mathbf u = 0,

where :math:`\nu_\mathrm{eff}` is the effective viscosity in the porous medium,
and :math:`K`, its permeability. It is also common to use the effective dynamic
viscosity :math:`\mu_\mathrm{eff} = \nu_\mathrm{eff} \rho`.

The inverse permeability :math:`K^{-1}` can be provided directly in cases where
it is zero in some regions, i.e. liquid-only regions. It is common to take
:math:`\nu_\mathrm{eff}=\nu` for simplicity (the default here).

The nondimensional implementation currently assumes :math:`\nu_\mathrm{eff}=\nu`
and :math:`K^{-1}>0`, such that


.. math::

   - \frac{1}{\mathrm{Re}}\nabla^2 \mathbf u + \mathbf u \cdot \nabla \mathbf u + \nabla p + \frac{1}{\mathrm{Re}\mathrm{Da}} \mathbf u = 0,

where all quantities have been nondimensionalized and
:math:`\mathrm{Da}=\frac{K}{L^2}` is the Darcy number.

In addition to the parameters provided for Navier-Stokes, the physical
parameters below must be provided.
For the dimensional version, the user must also pass the following keys:

* ``"permeability"`` or ``"inverse permeability"``: :math:`K` and :math:`K^{-1}`, respectively

* Optional: ``"effective kinematic viscosity"`` or ``"effective dynamic viscosity"``: :math:`\nu_\mathrm{eff}` and :math:`\mu_\mathrm{eff}`, respectively

To use the nondimensional version, the user must pass the following key:

* ``"Darcy number"``: :math:`\mathrm{Da}`

Boundary conditions
-------------------

A :py:class:`dict` containing boundary markers is passed when creating a
``FlowSolver`` object. Below are the different options for ``boundary_markers``
keys.

* ``"no slip"``: Sets velocity to zero, :math:`\mathbf u = 0`.

* ``"inlet velocity"``: Sets inlet velocity, :math:`\mathbf u = \mathbf
  u_\mathrm{in}`, which is passed in ``fluid_params`` as ``"inlet velocity"``.

* ``"outlet velocity"``: Sets outlet velocity, :math:`\mathbf u = \mathbf
  u_\mathrm{out}`, which is passed in ``fluid_params`` as ``"outlet
  velocity"``.

* ``"inlet pressure"``: Sets an inlet pressure traction using
  :math:`p_\mathrm{in}`, which is passed in ``fluid_params`` as
  ``"inlet pressure"``.

* ``"outlet pressure"``: Sets an outlet pressure traction using
  :math:`p_\mathrm{out}`, which is passed in ``fluid_params`` as
  ``"outlet pressure"``.

Numerical considerations
------------------------

The discretization is done using Taylor-Hood elements :cite:`Taylor1973`
(piecewise linear pressure, piecewise quadratic velocity), which satisfy the
inf-sup condition :cite:`Ladyzhenskaya1963,Brezzi1974,Babuvska1971`.

To achieve convergence other than at a low Reynolds number, it may be required
to do continuation on the parameter so that Newton's method has better initial
guesses. This can be done for example, by passing the Reynolds number as a
:class:`firedrake.constant.Constant`, and assigning a larger value after each
solve.

The highest Reynolds number that can be provided is dictated by the validity of
the steady-sate assumption. Indeed, as the Reynolds number increases, steady
flow becomes unstable. The critical Reynolds number at which transient effects
occur is problem dependent, typically somewhere within
:math:`10^2<\mathrm{Re}<10^4`.

For the Darcy number, using a very small value (:math:`\mathrm{Da}\sim
10^{-6}`) can be used to simulate a nearly impermeable wall, which is commonly
done in topology optimization. For smaller values, convergence issues can
arise. Reversly, as :math:`\mathrm{Da}\to\infty`, we simply recover
Navier-Stokes.

.. rubric:: References

.. bibliography:: ../_static/references.bib
   :filter: docname in docnames
