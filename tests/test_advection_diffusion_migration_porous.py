import pytest
from firedrake import *
from echemfem import EchemSolver


class AdvectionDiffusionMigrationSolver(EchemSolver):
    def __init__(self, N, vavg):
        mesh = UnitSquareMesh(N, N, quadrilateral=True)
        conc_params = []

        x, y = SpatialCoordinate(mesh)
        C1ex = cos(x) + sin(y) + 3
        C2ex = C1ex
        U1ex = sin(x) + cos(y) + 3
        U2ex = sin(x) + cos(y) + 3
        self.C1ex = C1ex
        self.C2ex = C2ex
        self.U1ex = U1ex
        self.U2ex = U2ex

        def f(C):
            D1 = 0.5
            D2 = 1.0
            z1 = 2.0
            z2 = -2.0
            K = 1.0
            f1 = div((self.vel - self.effective_diffusion(D1) * z1 * grad(U1ex)) * C1ex) - \
                div(self.effective_diffusion(D1) * grad(C1ex))
            f2 = div((self.vel - self.effective_diffusion(D2) * z2 * grad(U1ex)) * C2ex) - \
                div(self.effective_diffusion(D2) * grad(C2ex))
            f3 = div(- self.effective_diffusion(K, phase="solid") * grad(U2ex))
            return [f1, f2, f3]

        conc_params.append({"name": "C1",
                            "diffusion coefficient": 0.5,
                            "z": 2.,
                            "bulk": C1ex,
                            })

        conc_params.append({"name": "C2",
                            "diffusion coefficient": 1.0,
                            "z": -2.,
                            "bulk": C2ex,
                            })
        physical_params = {"flow": ["diffusion", "advection", "migration", "electroneutrality", "porous"],
                           "F": 1.0,  # C/mol
                           "R": 1.0,  # J/K/mol
                           "T": 1.0,  # K
                           "U_app": U1ex,  # V
                           "bulk reaction": f,
                           "v_avg": vavg,
                           "porosity": 0.5,
                           "solid conductivity": 1.,
                           "specific surface area": 1.,
                           "standard potential": 0.,
                           }

        super().__init__(conc_params, physical_params, mesh)

    def set_boundary_markers(self):
        self.boundary_markers = {"bulk dirichlet": (1, 2, 3, 4,),
                                 "applied": (1, 2, 3, 4,),
                                 "liquid applied": (1, 2, 3, 4),
                                 }

    def set_velocity(self):
        _, y = SpatialCoordinate(self.mesh)
        h = 1.0
        self.vel = as_vector(
            (6. * self.physical_params["v_avg"] / h**2 * y * (h - y), Constant(0.)))


def test_convergence_low_peclet():
    err_old = 1e6
    for i in range(6):
        solver = AdvectionDiffusionMigrationSolver(2**(i + 1), 1.0)
        solver.setup_solver()
        solver.solve()
        c1, U1, U2 = solver.u.subfunctions
        err = errornorm(solver.C1ex, c1) + \
            errornorm(solver.U1ex, U1) + errornorm(solver.U2ex, U2)
        assert err < 0.29 * err_old
        err_old = err


def test_convergence_high_peclet():
    err_old = 1e6
    for i in range(3, 6):
        solver = AdvectionDiffusionMigrationSolver(2**(i + 1), 1e3)
        solver.setup_solver()
        solver.solve()
        c1, U1, U2 = solver.u.subfunctions
        err = errornorm(solver.C1ex, c1) + \
            errornorm(solver.U1ex, U1) + errornorm(solver.U2ex, U2)
        assert err < 0.29 * err_old
        err_old = err
