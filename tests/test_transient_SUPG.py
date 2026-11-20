"""Exactly represented transient solutions require time in the SUPG residual."""

import pytest
from firedrake import (
    Constant, SpatialCoordinate, UnitSquareMesh, as_vector, assemble, errornorm,
)
from echemfem import TransientEchemSolver


class TransientTransportSolver(TransientEchemSolver):
    def __init__(self, transport):
        mesh = UnitSquareMesh(4, 4, quadrilateral=True)
        x, _ = SpatialCoordinate(mesh)
        self.time = Constant(0.)
        self.exact = (1 + self.time) * (2 + x)
        self.exact_potential = -100 * x
        self.migration = transport in ("migration", "advection_migration")
        self.advection_speed = 0. if transport == "stagnant" else 2.
        diffusion = .01
        charges = (1., -1., 0.)
        concentrations = [
            {"name": f"C{i}", "bulk": self.exact,
             "diffusion coefficient": diffusion, "z": charge}
            for i, charge in enumerate(charges)
        ]
        flow = ["diffusion"]
        if transport != "migration":
            flow.append("advection")
        advective_speed = self.advection_speed if "advection" in flow else 0.
        speeds = [advective_speed + (charge if self.migration else 0.)
                  for charge in charges]

        def sources(_):
            # C_t + beta.C_x - D*laplacian(C); beta is constant, laplacian=0.
            values = [2 + x + (1 + self.time) * speed for speed in speeds]
            if self.migration:
                # Opposite charges have equal concentrations; U is harmonic.
                values.append(0.)
            return values

        parameters = {"flow": flow, "bulk reaction": sources}
        if self.migration:
            flow.extend(("migration", "poisson"))
            parameters.update({"F": 1., "R": 1., "T": 1.,
                               "vacuum permittivity": 1.,
                               "relative permittivity": 1.,
                               "U_app": self.exact_potential})
        super().__init__(concentrations, parameters, mesh, family="CG", SUPG=True)
        self.save_solutions = False

    def set_boundary_markers(self):
        self.boundary_markers = {"bulk dirichlet": (1, 2, 3, 4)}
        if self.migration:
            self.boundary_markers["applied"] = (1, 2, 3, 4)
        else:
            self.boundary_markers.update({"inlet": (1, 2, 3, 4),
                                          "outlet": (1, 2, 3, 4)})

    def set_velocity(self):
        self.vel = as_vector((Constant(self.advection_speed), Constant(0.)))

    def init_solver_parameters(self):
        self.solver_parameters = {
            "snes_type": "newtonls", "snes_rtol": 1e-11, "snes_atol": 1e-11,
            "mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu",
        }
        self.potential_solver_parameters = self.solver_parameters.copy()

    def interpolate_exact(self, state):
        for field in state.subfunctions[:self.num_liquid]:
            field.interpolate(self.exact)
        if self.migration:
            state.subfunctions[self.i_Ul].interpolate(self.exact_potential)


@pytest.mark.parametrize("transport", ("advection", "migration", "advection_migration", "stagnant"))
def test_exact_transient_supg_solution(transport):
    solver = TransientTransportSolver(transport)
    solver.interpolate_exact(solver.u_old)
    solver.time.assign(.5)
    solver.dt.assign(.5)
    solver.interpolate_exact(solver.u)
    residual = assemble(solver.Form)
    for bc in solver.bcs:
        bc.zero(residual)
    with residual.dat.vec_ro as vector:
        assert vector.norm() < 1e-10

    solver.time.assign(0.)
    solver.interpolate_exact(solver.u)
    solver.setup_solver(initial_guess=False, initial_solve=False)
    solver.solve((0., .2, .5))
    for field in solver.u.subfunctions[:solver.num_liquid]:
        assert errornorm(solver.exact, field) < 1e-9
