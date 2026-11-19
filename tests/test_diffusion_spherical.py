import pytest
from firedrake import *
from echemfem import EchemSolver


class DiffusionSolver(EchemSolver):
    def __init__(self, N, family="DG"):
        mesh = UnitSquareMesh(N, N, quadrilateral=True)
        conc_params = []

        r, theta = SpatialCoordinate(mesh)
        C1ex = r**2 * (1 + cos(theta))
        C2ex = r**2 * (cos(theta) + cos(2 * theta))
        # need dC/dr = 0 at r=0 and dC/dtheta = 0 at theta=0
        self.C1ex = C1ex
        self.C2ex = C2ex

        def f(C):
            f1 = -6 - 6 * cos(theta) + 2 * r**2 * cos(theta)
            f2 = (-6 * (cos(theta) + cos(2 * theta))
                  + r**2 * (2 * cos(theta) + 4 * cos(2 * theta)
                            + 4 * cos(theta)**2))
            return [f1, f2]

        conc_params.append({"name": "C1",
                            "diffusion coefficient": 1.0,
                            "bulk": C1ex,
                            })

        conc_params.append({"name": "C2",
                            "diffusion coefficient": 1.0,
                            "bulk": C2ex,
                            })
        physical_params = {"flow": ["diffusion"],
                           "bulk reaction": f,
                           }

        super().__init__(conc_params, physical_params, mesh, family=family, coordinates="spherical")

    def set_boundary_markers(self):
        self.boundary_markers = {"bulk dirichlet": (2, 3, 4),
                                 }


def test_convergence():
    err_old = 1e6
    for i in range(5):
        solver = DiffusionSolver(2**(i + 1))
        solver.setup_solver()
        solver.solve()
        c1, c2 = solver.u.subfunctions
        err = errornorm(solver.C1ex, c1) + errornorm(solver.C2ex, c2)
        assert err < 0.36 * err_old
        err_old = err


def test_convergence_CG():
    err_old = 1e6
    for i in range(5):
        solver = DiffusionSolver(2**(i + 1), family="CG")
        solver.setup_solver()
        solver.solve()
        c1, c2 = solver.u.subfunctions
        err = errornorm(solver.C1ex, c1) + errornorm(solver.C2ex, c2)
        assert err < 0.36 * err_old
        err_old = err
