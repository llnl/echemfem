"""Accumulation must retain polynomial modes and the coordinate volume weight."""

import pytest
from firedrake import Constant, Function, SpatialCoordinate, UnitSquareMesh, assemble
from echemfem import TransientEchemSolver


class AccumulationSolver(TransientEchemSolver):
    def __init__(self, p=1, family="CG", cylindrical=False):
        mesh = UnitSquareMesh(1, 1, quadrilateral=True)
        mesh.coordinates.dat.data[:, 0] += 1.
        concentrations = [{"name": "C", "bulk": 1.,
                           "diffusion coefficient": 1.}]
        parameters = {"flow": ["diffusion"],
                      "bulk reaction": lambda _: (Constant(1.),)}
        super().__init__(concentrations, parameters, mesh, p=p,
                         family=family, cylindrical=cylindrical)
        self.save_solutions = False

    def set_boundary_markers(self):
        self.boundary_markers = {}


@pytest.mark.parametrize("family", ("CG", "DG"))
@pytest.mark.parametrize("cylindrical", (False, True))
@pytest.mark.parametrize("p,expected", ((1, 1 / 12), (2, 1 / 180), (3, 1 / 2800)))
def test_polynomial_mass(p, expected, cylindrical, family):
    solver = AccumulationSolver(p=p, family=family, cylindrical=cylindrical)
    x, _ = SpatialCoordinate(solver.mesh)
    s = x - 1.5
    polynomial = {1: s, 2: s**2 - 1 / 12, 3: s**3 - 3 * s / 20}[p]
    field = Function(solver.V).interpolate(polynomial)
    solver.dt.assign(1.)
    form, _ = solver.mass_accumulation_form(
        field, 0, field, solver.conc_params[0])
    # Squared Legendre modes have these analytic integrals. With the radial
    # weight x on [1,2], symmetry multiplies every integral by exactly 3/2.
    if cylindrical:
        expected *= 1.5
    assert assemble(form) == pytest.approx(expected, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("family", ("CG", "DG"))
def test_cylindrical_uniform_source_keeps_concentration_uniform(family):
    solver = AccumulationSolver(p=2, family=family, cylindrical=True)
    solver.setup_solver(initial_solve=False)
    solver.solve((0., .25, .5, 1.))
    # Closed boundaries and a uniform unit source give C(t) = 1 + t.
    assert solver.u.subfunctions[0].dat.data_ro == pytest.approx(2., abs=1e-11)
