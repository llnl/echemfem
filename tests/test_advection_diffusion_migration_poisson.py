import pytest
from firedrake import *
from echemfem import EchemSolver


class DiffusionMigrationSolver(EchemSolver):
    def __init__(self, N):
        # Create an initial coarse mesh
        initial_mesh = UnitSquareMesh(2, 2, quadrilateral=True)
        # Create a mesh hierarchy with uniform refinements
        hierarchy = MeshHierarchy(initial_mesh, N)
        # Use the finest mesh for the initial discretization
        mesh = hierarchy[-1]
        x, y = SpatialCoordinate(mesh)

        conc_params = []

        C1ex = cos(x) + sin(y) + 3
        C2ex = cos(x) - sin(y) + 3
        Uex = sin(x) + cos(y) + 3
        self.C1ex = C1ex
        self.C2ex = C2ex
        self.Uex = Uex

        def f(C):
            D1 = 0.5
            D2 = 1.0
            z1 = 2.0
            z2 = -2.0
            f1 = div((self.vel - D1 * z1 * grad(Uex))
                     * C1ex) - div(D1 * grad(C1ex))
            f2 = div((self.vel - D2 * z2 * grad(Uex))
                     * C2ex) - div(D2 * grad(C2ex))
            f3 = div((- D1 * z1**2 * grad(Uex)) * C1ex) + \
                div((- D2 * z2**2 * grad(Uex)) * C2ex)
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
        physical_params = {
            "flow": [
                "diffusion",
                "advection",
                "migration",
                "poisson"],
            "F": 1.0,
            "R": 1.0,
            "T": 1.0,
            "U_app": Uex,
            "bulk reaction": f,
            "v_avg": 1.,
        }

        super().__init__(conc_params, physical_params, mesh)

    def set_boundary_markers(self):
        self.boundary_markers = {"bulk dirichlet": (1, 2, 3, 4,),
                                 "applied": (1, 2, 3, 4,),
                                 }

    def set_velocity(self):
        if self.mesh.geometric_dimension() == 3:
            _, y, _ = SpatialCoordinate(self.mesh)
        else:
            _, y = SpatialCoordinate(self.mesh)

        h = 1.0
        x_vel = 6. * self.physical_params["v_avg"] / h**2 * y * (h - y)

        if self.mesh.geometric_dimension() == 3:
            self.vel = as_vector(
                (x_vel, Constant(0.), Constant(0.)))
        else:
            self.vel = as_vector(
                (x_vel, Constant(0.)))


def test_convergence():
    err_old = 1e6
    for i in range(3):
        solver = DiffusionMigrationSolver(i+1)
        solver.setup_solver()
        solver.solve()
        c1, c2, U = solver.u.subfunctions
        err = errornorm(solver.C1ex, c1) + errornorm(solver.C2ex, c2)\
            + errornorm(solver.Uex, U)
        assert err < 0.29 * err_old
        err_old = err


