API
===

EchemSolver classes
-------------------

.. autoclass:: echemfem.EchemSolver
   :members:

.. autoclass:: echemfem.TransientEchemSolver
   :members:

Transient accumulation uses the coordinate volume weight and automatic
quadrature based on the complete integrand. This retains higher-order mass
terms without changing the quadrature used for steady forms. In custom
forms, ``self.dx(degree=None)`` selects automatic quadrature, while
``self.dx(degree=k)`` requests degree ``k`` explicitly.

For CG with SUPG enabled, time accumulation is included in the stabilization
using the same transport velocity and stabilization coefficient as the steady
species equation. In spherical coordinates, the streamline derivative uses
the physical gradient. DG accumulation does not add SUPG terms.

Coordinate measures
-------------------

Use ``coordinates="cylindrical"`` for an axisymmetric ``(r, z)`` mesh or
``coordinates="spherical"`` for an axisymmetric ``(r, theta)`` mesh, where
``theta`` is the polar angle in radians. Both coordinate systems integrate per
unit azimuthal angle. Multiply integrated quantities by :math:`2\pi` when a
full rotational volume, area, or rate is needed.

For spherical meshes, velocities use physical radial and polar components,
and custom Neumann data and surface reaction rates are per unit physical
surface area. Use :meth:`echemfem.EchemSolver.grad` and
:meth:`echemfem.EchemSolver.div` in user-defined spherical source terms rather
than the Cartesian Firedrake operators. In particular,
:math:`\nabla C=(\partial_r C,\partial_\theta C/r)`.

The spherical cell measure applies :math:`r^2\sin\theta`. Facet measures
include the physical surface Jacobian, which is :math:`r^2\sin\theta` on a
constant-radius face and :math:`r\sin\theta` on a constant-angle face.
The origin and polar axes are coordinate singularities; regular solutions
must satisfy the corresponding symmetry conditions.

.. note::

   The inverse-metric guards in ``grad()`` and ``div()`` prevent division by
   zero but do not compute analytic limits at coordinate singularities. This
   applies at the spherical origin and poles, and at the cylindrical axis
   for ``div()``.

   With the default quadrature, the solver evaluates volume terms at interior
   quadrature points. User-defined ``bulk reaction`` source terms remain UFL
   expressions and are integrated directly, so their values exactly at the
   singular boundaries are not sampled. Meshes may include those boundaries.

   Before point evaluation or nodal interpolation of these expressions on a
   singular boundary, supply the appropriate analytic limiting expression.
   For example, the regular spherical vector ``(r, 0)`` has divergence ``3``
   at the origin; the inverse-radius guard does not supply that limit.

.. autoclass:: echemfem.CylindricalMeasure

.. autoclass:: echemfem.SphericalMeasure

FlowSolver classes
-------------------

.. autoclass:: echemfem.FlowSolver
   :members:

.. autoclass:: echemfem.NavierStokesFlowSolver
   :members:

.. autoclass:: echemfem.NavierStokesBrinkmanFlowSolver
   :members:

Utility meshes
--------------
These are boundary layer meshes, which are modified versions of :func:`firedrake.utility_meshes.IntervalMesh` and :func:`firedrake.utility_meshes.RectangleMesh`.

.. automodule:: echemfem.utility_meshes
   :members:

.. toctree::
   :hidden:
