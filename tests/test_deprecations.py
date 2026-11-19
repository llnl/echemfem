import pytest
from firedrake import *
from echemfem import EchemSolver

with pytest.deprecated_call(
        match="cylindrical is deprecated; use coordinates='cylindrical' instead."):
    class DiffusionSolver(EchemSolver):
        def __init__(self, N, family="DG"):
            mesh = UnitSquareMesh(N, N, quadrilateral=True)
            conc_params = []

            r, z = SpatialCoordinate(mesh)
            C1ex = sin(z) + cos(r)
            C2ex = cos(z) * cos(r) + sin(z) * cos(r)
            # need dC/dr = 0 at r=0
            self.C1ex = C1ex
            self.C2ex = C2ex

            def f(C):
                f1 = sin(r)/r + cos(r) + sin(z)
                f2 = (r * cos(r) + sin(r)) * (cos(z) + sin(z))/r + cos(r) * (cos(z) + sin(z))
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

            super().__init__(conc_params, physical_params, mesh, family=family, cylindrical=True)

        def set_boundary_markers(self):
            self.boundary_markers = {"bulk dirichlet": (2, 3, 4),
                                        }

    solver = DiffusionSolver(N=1)
