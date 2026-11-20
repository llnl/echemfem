"""Manufactured transport problems derived in physical spherical components."""

import pytest
from firedrake import (
    Constant, SpatialCoordinate, UnitSquareMesh, as_vector, assemble, cos, errornorm,
    inner, sin,
)
from echemfem import EchemSolver, TransientEchemSolver


def _shell_mesh(n):
    mesh = UnitSquareMesh(n, n, quadrilateral=True)
    mesh.coordinates.dat.data[:, 0] += 1.0
    mesh.coordinates.dat.data[:, 1] *= 0.9
    mesh.coordinates.dat.data[:, 1] += 0.3
    return mesh


def _spherical_gradient(field, r):
    return as_vector((field.dx(0), field.dx(1) / r))


def _spherical_divergence(flux, r, theta):
    # Independent strong form: flux components are physical radial/polar ones.
    return ((r**2 * flux[0]).dx(0) / r**2
            + (sin(theta) * flux[1]).dx(1) / (r * sin(theta)))


class SphericalTransportSolver(EchemSolver):
    def __init__(self, n, family, migration=False, supg=False,
                 diffusion_scale=1.0, velocity_scale=1.0):
        mesh = _shell_mesh(n)
        r, theta = SpatialCoordinate(mesh)
        self.migration = migration
        self.velocity_scale = velocity_scale
        self.exact_concentrations = (
            3.0 + 0.2 * cos(theta) + 0.1 * sin(r),
            3.0 - 0.2 * cos(theta) + 0.1 * cos(r),
        )
        self.exact_potential = 0.2 * sin(r) + 0.1 * cos(theta)
        diffusion = (0.5 * diffusion_scale, diffusion_scale)
        charges = (1.0, -1.0) if migration else (0.0, 0.0)
        concentrations = [
            {"name": f"C{i + 1}", "bulk": exact,
             "diffusion coefficient": d, "z": z}
            for i, (exact, d, z) in enumerate(
                zip(self.exact_concentrations, diffusion, charges))
        ]
        flow = ["diffusion", "advection"]
        physical_params = {"flow": flow}
        if migration:
            flow.extend(("migration", "poisson"))
            physical_params.update({
                "F": 1.0, "R": 1.0, "T": 1.0,
                "vacuum permittivity": 0.2,
                "relative permittivity": 1.0,
                "U_app": self.exact_potential,
            })

        def sources(_):
            potential_grad = _spherical_gradient(self.exact_potential, r)
            values = [
                _spherical_divergence(
                    exact * (self.vel - d * z * potential_grad)
                    - d * _spherical_gradient(exact, r), r, theta)
                for exact, d, z in zip(
                    self.exact_concentrations, diffusion, charges)
            ]
            if migration:
                values.append(
                    _spherical_divergence(-0.2 * potential_grad, r, theta)
                    - sum(z * exact for z, exact in zip(
                        charges, self.exact_concentrations)))
            return values

        physical_params["bulk reaction"] = sources
        super().__init__(concentrations, physical_params, mesh,
                         coordinates="spherical", family=family, SUPG=supg)
        self.save_solutions = False

    def set_boundary_markers(self):
        self.boundary_markers = {
            "bulk dirichlet": (1, 2, 3),
            "neumann": (4,),
            "inlet": (1, 2, 3, 4),
            "outlet": (1, 2, 3, 4),
        }
        if self.migration:
            self.boundary_markers["applied"] = (1, 2, 3, 4)

    def set_velocity(self):
        r, theta = SpatialCoordinate(self.mesh)
        self.vel = self.velocity_scale * as_vector(
            (0.5 / r**2, 0.15 * sin(theta)))

    def neumann(self, concentration, conc_params, state):
        r, _ = SpatialCoordinate(self.mesh)
        # Marker 4 is constant theta: the physical unit normal is e_theta.
        index = int(conc_params["name"][1:]) - 1
        return (conc_params["diffusion coefficient"]
                * self.exact_concentrations[index].dx(1) / r)

    def init_solver_parameters(self):
        # A direct solve makes convergence evidence independent of preconditioning.
        self.solver_parameters = {
            "snes_type": "newtonls", "snes_linesearch_type": "l2",
            "snes_rtol": 1e-10, "snes_atol": 1e-11,
            "snes_max_it": 20, "mat_type": "aij",
            "ksp_type": "preonly", "pc_type": "lu",
        }
        self.potential_solver_parameters = self.solver_parameters.copy()


@pytest.mark.parametrize("family", ("CG", "DG"))
@pytest.mark.parametrize("migration", (False, True))
def test_angular_transport_convergence(family, migration):
    previous = None
    for n in (4, 8, 16):
        solver = SphericalTransportSolver(n, family, migration=migration)
        solver.setup_solver(initial_solve=False)
        solver.solve()
        exact = solver.exact_concentrations
        if migration:
            exact += (solver.exact_potential,)
        error = sum(errornorm(reference, numerical, norm_type="L2")
                    for reference, numerical in zip(exact, solver.u.subfunctions))
        if previous is not None:
            assert error < 0.36 * previous
        previous = error

def test_darcy_velocity_uses_physical_polar_derivative():
    solver = SphericalTransportSolver(4, "CG")
    r, theta = SpatialCoordinate(solver.mesh)
    solver.pl = r**2 * cos(theta)
    solver.Sw = 0.7
    solver.flow["darcy"] = True
    solver.physical_params.update({
        "permeability": 0.5, "liquid viscosity": 0.2,
    })
    solver.set_darcy_velocity()
    mobility = 0.5 / 0.2 * 0.7**3
    reference = mobility * as_vector((-2 * r * cos(theta), r * sin(theta)))
    discrepancy = solver.vel - reference
    assert assemble(inner(discrepancy, discrepancy)
                    * solver.dx(domain=solver.mesh)) < 1e-24


def test_high_peclet_supg_convergence():
    # Small diffusivity keeps stabilization active as the mesh is refined.
    # The prescribed polar velocity has nonzero spherical divergence, so this
    # also checks the conservative SUPG residual, not only directional gradients.
    previous = None
    for n in (8, 16, 32):
        solver = SphericalTransportSolver(
            n, "CG", supg=True, diffusion_scale=0.02, velocity_scale=8.0)
        solver.setup_solver(initial_solve=False)
        solver.solve()
        error = sum(errornorm(reference, numerical, norm_type="L2")
                    for reference, numerical in zip(
                        solver.exact_concentrations, solver.u.subfunctions))
        if previous is not None:
            assert error < 0.4 * previous
        previous = error


class SphericalTransientSolver(TransientEchemSolver):
    def __init__(self, n, family):
        mesh = _shell_mesh(n)
        r, theta = SpatialCoordinate(mesh)
        self.time = Constant(0.0)
        spatial_factor = r**2 * (1 + cos(theta))
        self.exact = (1 + self.time) * spatial_factor
        # dC/dt - laplacian(C); linear time is represented exactly by BE.
        source = spatial_factor + (1 + self.time) * (-6 - 4 * cos(theta))
        concentrations = [{"name": "C", "bulk": self.exact,
                           "diffusion coefficient": 1.0}]
        physical_params = {"flow": ["diffusion"],
                           "bulk reaction": lambda _: (source,)}
        super().__init__(concentrations, physical_params, mesh,
                         coordinates="spherical", family=family)
        self.save_solutions = False

    def set_boundary_markers(self):
        self.boundary_markers = {"bulk dirichlet": (1, 2, 3, 4)}


@pytest.mark.parametrize("family", ("CG", "DG"))
def test_transient_spherical_mass_convergence(family):
    previous = None
    for n in (4, 8, 16):
        solver = SphericalTransientSolver(n, family)
        solver.u.subfunctions[0].interpolate(solver.exact)
        solver.setup_solver(initial_guess=False)
        solver.solve((0.0, 0.5, 1.0))
        error = errornorm(solver.exact, solver.u.subfunctions[0], norm_type="L2")
        if previous is not None:
            assert error < 0.36 * previous
        previous = error
