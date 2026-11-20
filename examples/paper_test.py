import pytest
from firedrake import *
from echemfem import EchemSolver
from mpi4py import MPI
from petsc4py import PETSc
"""
Convergence tests for the DG scheme in
Roy, T., Andrej, J. and Beck, V.A., 2023. A scalable DG solver for the
electroneutral Nernst-Planck equations. Journal of Computational Physics, 475,
p.111859.
"""

PETSc.Sys.popErrorHandler()

# set_log_level(DEBUG)
GLOBAL_N_RANKS = MPI.Comm.Get_size(MPI.COMM_WORLD)
if GLOBAL_N_RANKS % 8 == 0:
    REDFACTOR = int(GLOBAL_N_RANKS / 8)
else:
    REDFACTOR = 1


class DiffusionMigrationSolver(EchemSolver):
    def __init__(self, N, extruded=False, gmg=False):
        if extruded and gmg:
            N = int(N / 2)
            plane_mesh = UnitSquareMesh(N, N, quadrilateral=True)
            plane_mesh_hierarchy = MeshHierarchy(plane_mesh, 2)
            extruded_hierarchy = ExtrudedMeshHierarchy(
                plane_mesh_hierarchy, 1.0, N)
            mesh = extruded_hierarchy[-1]
            x, y, _ = SpatialCoordinate(mesh)
        elif extruded:
            plane_mesh = UnitSquareMesh(N, N, quadrilateral=True)
            mesh = ExtrudedMesh(plane_mesh, N, 1.0 / N)
            x, y, _ = SpatialCoordinate(mesh)
        else:
            # Create an initial coarse mesh
            initial_mesh = UnitSquareMesh(2, 2, quadrilateral=True)
            # Create a mesh hierarchy with uniform refinements
            hierarchy = MeshHierarchy(initial_mesh, N)
            # Use the finest mesh for the initial discretization
            mesh = hierarchy[-1]
            x, y = SpatialCoordinate(mesh)

        conc_params = []

        C1ex = cos(x) + sin(y) + 3
        C2ex = cos(x) + sin(y) + 3
        Uex = sin(x) + cos(y) + 3
        self.C1ex = C1ex
        self.C2ex = C2ex
        self.Uex = Uex

        D1 = 0.5e-5
        D2 = 1.0e-5
        z1 = 2.0
        z2 = -2.0

        def f(C):
            f1 = div((self.vel - D1 * z1 * grad(Uex))
                     * C1ex) - div(D1 * grad(C1ex))
            f2 = div((self.vel - D2 * z2 * grad(Uex))
                     * C2ex) - div(D2 * grad(C2ex))
            return [f1, f2]

        conc_params.append({"name": "C1",
                            "diffusion coefficient": D1,
                            "z": z1,
                            "bulk": C1ex,
                            })

        conc_params.append({"name": "C2",
                            "diffusion coefficient": D2,
                            "z": z2,
                            "bulk": C2ex,
                            })
        physical_params = {
            "flow": [
                "diffusion",
                "advection",
                "migration",
                "electroneutrality"],
            "F": 1.0,
            "R": 1.0,
            "T": 1.0,
            "U_app": Uex,
            "bulk reaction": f,
            "v_avg": 1.0,
        }

        super().__init__(conc_params, physical_params, mesh, p=1)
        # Select a geometric multigrid preconditioner to make use of the
        # MeshHierarchy
        U_is = self.num_mass
        is_list = [str(i) for i in range(self.num_mass)]
        C_is = ",".join(is_list)

        self.init_solver_parameters(
            custom_solver={
                # "snes_view": None,
                # "snes_converged_reason": None,
                # "snes_monitor": None,
                "snes_rtol": 1e-10,
                # "ksp_monitor": None,
                # "ksp_converged_reason": None,
                "ksp_type": "fgmres",
                "ksp_rtol": 1e-3,
                "pc_type": "fieldsplit",
                "pc_fieldsplit_0_fields": U_is,
                "pc_fieldsplit_1_fields": C_is,
                "fieldsplit_0": {
                    # "ksp_converged_reason": None,
                    "ksp_rtol": 1e-1,
                    "ksp_type": "cg",
                    "pc_type": "hypre",
                    "pc_hypre_boomeramg": {
                            "strong_threshold": 0.7,
                            "coarsen_type": "HMIS",
                            "agg_nl": 3,
                            "interp_type": "ext+i",
                            "agg_num_paths": 5,
                            # "print_statistics": None,
                    },
                },
                "fieldsplit_1": {
                    # "ksp_converged_reason": None,
                    "ksp_rtol": 1e-1,
                    "ksp_type": "gmres",
                    "pc_type": "mg",
                    "mg_levels_ksp_type": "richardson",
                    "mg_levels_pc_type": "bjacobi",
                    "mg_levels_sub_pc_type": "ilu",
                    "mg_coarse": {
                        "pc_type": "python",
                        "pc_python_type": "firedrake.AssembledPC",
                        "assembled": {
                                "mat_type": "aij",
                                "pc_type": "telescope",
                                "pc_telescope_reduction_factor": REDFACTOR,
                                "pc_telescope_subcomm_type": "contiguous",
                                "telescope_pc_type": "lu",
                                "telescope_pc_factor_mat_solver_type": "mumps",
                        }
                    },
                },
            },
            custom_potential_solver={
                "mat_type": "aij",
                # "snes_view": None,
                # "snes_monitor": None,
                "snes_rtol": 1e-6,
                # "ksp_monitor": None,
                # "ksp_converged_reason": None,
                "ksp_type": "cg",
                "ksp_rtol": 1e-2,
                "pc_type": "hypre",
                "pc_hypre_boomeramg": {
                    "strong_threshold": 0.7,
                    "coarsen_type": "HMIS",
                    "agg_nl": 3,
                    "interp_type": "ext+i",
                    "agg_num_paths": 5,
                },
            })

    def set_boundary_markers(self):
        self.boundary_markers = {"bulk dirichlet": (1, 2, 3, 4,),
                                 "applied": (1, 2, 3, 4,),
                                 }

    def set_velocity(self):
        if self.mesh.geometric_dimension == 3:
            _, y, _ = SpatialCoordinate(self.mesh)
        else:
            _, y = SpatialCoordinate(self.mesh)

        h = 1.0
        x_vel = 6. * self.physical_params["v_avg"] / h**2 * y * (h - y)

        if self.mesh.geometric_dimension == 3:
            self.vel = as_vector(
                (x_vel, Constant(0.), Constant(0.)))
        else:
            self.vel = as_vector(
                (x_vel, Constant(0.)))


def test_convergence(extruded=False, gmg=False):
    err_old = 1e6
    for i in [2, 4, 8, 16, 32]:
        if gmg:
            solver = DiffusionMigrationSolver(i, extruded, gmg)
        else:
            solver = DiffusionMigrationSolver(i, extruded, gmg)
        solver.setup_solver()
        solver.solve()
        c1, U = solver.u.subfunctions
        errc1 = errornorm(solver.C1ex, c1)
        errU = errornorm(solver.Uex, U)
        err = errc1 + errU
        # only prints number of cells of the 2D mesh
        PETSc.Sys.Print("cells = {} L2err = {:.5E} C1err = {:.5E} Uerr = {:.5E}".format(
            solver.comm.allreduce(solver.mesh.cell_set.size, op=MPI.SUM), err, errc1, errU))
        err_old = err


test_convergence(extruded=True, gmg=True)
