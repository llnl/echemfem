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
species equation. DG accumulation does not add SUPG terms.

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
