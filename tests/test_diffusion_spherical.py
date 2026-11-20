import math

import pytest
from firedrake import *
from echemfem import EchemSolver


def spherical_rectangle(N, radial_range, polar_range):
    r0, r1 = radial_range
    theta0, theta1 = polar_range
    mesh = RectangleMesh(N, N, r1 - r0, theta1 - theta0,
                         quadrilateral=True)
    mesh.coordinates.dat.data[:, 0] += r0
    mesh.coordinates.dat.data[:, 1] += theta0
    return mesh


class DiffusionSolver(EchemSolver):
    def __init__(self, N, family="DG", domain="annulus", solution="angular",
                 angular_neumann=False):
        self.domain = domain
        self.angular_neumann = angular_neumann
        radial_range = (0., 1.) if domain == "origin_poles" else (1., 2.)
        polar_range = (0., math.pi) if domain == "origin_poles" else (.3, 1.3)
        mesh = spherical_rectangle(N, radial_range, polar_range)
        r, theta = SpatialCoordinate(mesh)

        if angular_neumann:
            # r*cos(theta) is the Cartesian z coordinate and is harmonic.
            self.exact = (r * cos(theta),)
            forcing = (Constant(0.),)
        elif solution == "radial":
            # 1/r is harmonic on an annulus, with zero angular flux.
            self.exact = (1 / r,)
            forcing = (Constant(0.),)
        else:
            self.exact = (r**2 * (1 + cos(theta)),
                          r**2 * (cos(theta) + cos(2 * theta)))
            # These are -laplacian(C) in physical spherical coordinates.
            forcing = (-6 - 4 * cos(theta), 2 - 4 * cos(theta))

        conc_params = [{"name": f"C{i}", "diffusion coefficient": 1.,
                        "bulk": exact}
                       for i, exact in enumerate(self.exact)]
        physical_params = {"flow": ["diffusion"],
                           "bulk reaction": lambda C: forcing}
        super().__init__(conc_params, physical_params, mesh, family=family,
                         coordinates="spherical")
        self.save_solutions = False

    def set_boundary_markers(self):
        if self.angular_neumann:
            self.boundary_markers = {"bulk dirichlet": (1, 2, 3),
                                     "neumann": (4,)}
        elif self.domain == "origin_poles":
            # Regularity supplies zero flux at r=0 and both poles.
            self.boundary_markers = {"bulk dirichlet": (2,)}
        else:
            self.boundary_markers = {"bulk dirichlet": (1, 2, 3, 4)}

    def neumann(self, C, conc_params, u):
        _, theta = SpatialCoordinate(self.mesh)
        # On the upper angular face the physical outward normal is e_theta.
        return -sin(theta)


def convergence_errors(**kwargs):
    errors = []
    for N in (4, 8, 16):
        solver = DiffusionSolver(N, **kwargs)
        solver.setup_solver()
        solver.solve()
        errors.append(sum(errornorm(exact, approximate)
                          for exact, approximate in
                          zip(solver.exact, solver.u.subfunctions)))
    return errors


@pytest.mark.parametrize("family", ("CG", "DG"))
@pytest.mark.parametrize("domain", ("annulus", "origin_poles"))
def test_angular_convergence(family, domain):
    errors = convergence_errors(family=family, domain=domain)
    assert errors[1] < .36 * errors[0]
    assert errors[2] < .36 * errors[1]


@pytest.mark.parametrize("family", ("CG", "DG"))
def test_radial_harmonic_convergence(family):
    errors = convergence_errors(family=family, solution="radial")
    assert errors[1] < .36 * errors[0]
    assert errors[2] < .36 * errors[1]


@pytest.mark.parametrize("family", ("CG", "DG"))
def test_physical_angular_neumann_flux(family):
    errors = convergence_errors(family=family, angular_neumann=True)
    assert errors[1] < .36 * errors[0]
    assert errors[2] < .36 * errors[1]
