import warnings

import pytest
from firedrake import *
from echemfem import EchemSolver


class DiffusionSolver(EchemSolver):
    def __init__(self, **kwargs):
        mesh = UnitSquareMesh(1, 1, quadrilateral=True)
        mesh.coordinates.dat.data[:, 0] += 1.
        concentrations = [{"name": "C", "diffusion coefficient": 1.,
                           "bulk": Constant(1.)}]
        super().__init__(concentrations, {"flow": ["diffusion"]}, mesh,
                         family="CG", **kwargs)
        self.save_solutions = False

    def set_boundary_markers(self):
        self.boundary_markers = {"bulk dirichlet": (1, 2, 3, 4)}


@pytest.mark.parametrize("coordinates", (None, "cylindrical"))
def test_cylindrical_flag_is_deprecated(coordinates):
    with pytest.deprecated_call(
            match="cylindrical is deprecated; use coordinates='cylindrical' instead."):
        solver = DiffusionSolver(cylindrical=True, coordinates=coordinates)
    assert assemble(1 * solver.dx(domain=solver.mesh)) == pytest.approx(1.5)


@pytest.mark.parametrize("coordinates", ("cartesian", "spherical"))
def test_cylindrical_flag_overrides_conflicting_coordinates(coordinates):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        solver = DiffusionSolver(cylindrical=True, coordinates=coordinates)
    assert any(issubclass(w.category, DeprecationWarning)
               and "cylindrical is deprecated" in str(w.message)
               for w in recorded)
    assert any(issubclass(w.category, UserWarning)
               and "Changing to cylindrical coordinates" in str(w.message)
               for w in recorded)
    assert assemble(1 * solver.dx(domain=solver.mesh)) == pytest.approx(1.5)


@pytest.mark.parametrize("coordinates", (None, "cartesian", "cylindrical"))
def test_coordinates_option_does_not_emit_legacy_warning(coordinates):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        DiffusionSolver(coordinates=coordinates)
    assert not any("cylindrical is deprecated" in str(w.message)
                   for w in recorded)
