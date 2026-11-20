"""Physical divergence accepts constant components and finite element fields."""

import pytest
from firedrake import (
    Constant, Function, SpatialCoordinate, UnitSquareMesh, VectorFunctionSpace,
    as_vector, assemble, cos, dx, sin,
)
from echemfem import EchemSolver


class DivergenceSolver(EchemSolver):
    def __init__(self, coordinates):
        # Stay away from coordinate singularities: their limits are a separate
        # concern from retaining the domain of constant vector components.
        mesh = UnitSquareMesh(2, 2, quadrilateral=True)
        mesh.coordinates.dat.data[:, 0] += 1.
        mesh.coordinates.dat.data[:, 1] *= .9
        mesh.coordinates.dat.data[:, 1] += .3
        super().__init__(
            [{"name": "C", "bulk": 1., "diffusion coefficient": 1.}],
            {"flow": ["diffusion"]}, mesh, family="CG",
            coordinates=coordinates)
        self.save_solutions = False

    def set_boundary_markers(self):
        self.boundary_markers = {}


@pytest.fixture(params=("cartesian", "cylindrical", "spherical"))
def solver(request):
    return DivergenceSolver(request.param)


def assert_matches(solver, actual, expected):
    error = actual - expected
    # UFL removes identically zero integrals, leaving no domain to assemble on.
    if error != 0:
        assert assemble(error**2 * dx(domain=solver.mesh, degree=10)) < 1e-24


@pytest.mark.parametrize("case", ("radial", "angular", "mixed", "constant", "zero"))
def test_literal_vector_components(solver, case):
    r, angle = SpatialCoordinate(solver.mesh)
    cotangent = cos(angle) / sin(angle)
    fields = {
        "radial": as_vector((r, 0)),
        "angular": as_vector((0, angle)),
        "mixed": as_vector((r, 2)),
        "constant": as_vector((2, -3)),
        "zero": as_vector((0, 0)),
    }
    # Analytic divergences in the order of the cases above. A vector with
    # constant physical components can still have nonzero physical divergence.
    expected = {
        "cartesian": {"radial": 1, "angular": 1, "mixed": 1,
                      "constant": 0, "zero": 0},
        "cylindrical": {"radial": 2, "angular": 1, "mixed": 2,
                        "constant": 2 / r, "zero": 0},
        "spherical": {"radial": 3, "angular": (1 + angle * cotangent) / r,
                      "mixed": 3 + 2 * cotangent / r,
                      "constant": (4 - 3 * cotangent) / r, "zero": 0},
    }
    assert_matches(solver, solver.div(fields[case]), expected[solver.coordinates][case])


def test_mutable_constant_vector(solver):
    r, angle = SpatialCoordinate(solver.mesh)
    cotangent = cos(angle) / sin(angle)
    field = Constant((2., -3.))
    divergence = solver.div(field)
    for values, expected in (
        ((2., -3.), {"cartesian": 0, "cylindrical": 2 / r,
                    "spherical": (4 - 3 * cotangent) / r}),
        ((0., 4.), {"cartesian": 0, "cylindrical": 0,
                   "spherical": 4 * cotangent / r}),
    ):
        field.assign(values)
        assert_matches(solver, divergence, expected[solver.coordinates])


@pytest.mark.parametrize("representation", ("expression", "function"))
def test_nonconstant_vector(solver, representation):
    r, angle = SpatialCoordinate(solver.mesh)
    field = as_vector((r**2 * angle, r * angle**2))
    if representation == "function":
        space = VectorFunctionSpace(solver.mesh, "CG", 2)
        field = Function(space).interpolate(field)
    expected = {
        "cartesian": 4 * r * angle,
        "cylindrical": 5 * r * angle,
        "spherical": 4 * r * angle + 2 * angle + angle**2 * cos(angle) / sin(angle),
    }
    assert_matches(solver, solver.div(field), expected[solver.coordinates])
