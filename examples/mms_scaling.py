from echemfem import EchemSolver
import argparse
from firedrake import *
from mpi4py import MPI
from firedrake.petsc import PETSc
from firedrake import AssembledPC
PETSc.Sys.popErrorHandler()

set_log_level(DEBUG)

"""
A manufactured example of 3D electroneutral Nernst-Planck to evaluate the
convergence of the DG scheme and scaling of custom block preconditioners. The
file submitter_mms.pl can be adapted to launch slurm jobs using perl.

DG scheme and preconditioning from:
Roy, T., Andrej, J. and Beck, V.A., 2023. A scalable DG solver for the
electroneutral Nernst-Planck equations. Journal of Computational Physics, 475,
p.111859.
"""

parser = argparse.ArgumentParser()
parser.add_argument('-ref_levels', type=int, default=0)
parser.add_argument('-stats_file', type=str, default='')
parser.add_argument('-overwrite_stats_file', type=bool, default=False)
parser.add_argument('-csolver', type=str, default='asm')
parser.add_argument('-degree', type=int, default=1)
args, unknown = parser.parse_known_args()

GLOBAL_N_RANKS = MPI.Comm.Get_size(MPI.COMM_WORLD)
if GLOBAL_N_RANKS % 8 == 0:
    REDFACTOR = int(GLOBAL_N_RANKS / 8)
else:
    REDFACTOR = 1

p = Constant(args.degree)


class CellIntegralPC(AssembledPC):
    def form(self, pc, test, trial):
        a, bcs = super().form(pc, test, trial)
        return Form(a.integrals_by_type("cell")), bcs


class Q1PC(PMGPC):
    """
    p-multigrid coarsening with Lagrange elements
    """

    def coarsen_element(self, ele):
        k = super().max_degree(ele)
        if k <= self.coarse_degree:
            raise ValueError
        family = {e.family() for e in ele.sub_elements()} or {ele.family()}
        # if family <= {"Q", "Lagrange"}:
        if family <= {"Q", "Lagrange"} or True:
            k = self.coarse_degree
        e = FiniteElement("Lagrange", cell=ele.cell(), degree=k, variant="fdm")
        return TensorElement(e, shape=ele.value_shape()) if ele.value_shape() else e

    def coarsen_form(self, form, args):
        """recover CG formulation from IP-DG by discarding facet terms"""
        return super(Q1PC, self).coarsen_form(Form(form.integrals_by_type("cell")), args)


class BortelsSolver(EchemSolver):
    def __init__(self):
        if args.degree == 1:
            N = 16
        else:
            N = 8
        plane_mesh = UnitSquareMesh(N, N, quadrilateral=True)
        plane_mesh_hierarchy = MeshHierarchy(plane_mesh, args.ref_levels)
        extruded_hierarchy = ExtrudedMeshHierarchy(
            plane_mesh_hierarchy, 1.0, N)
        mesh = extruded_hierarchy[-1]
        x, y, _ = SpatialCoordinate(mesh)

        self.csolver = args.csolver

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

        super().__init__(
            conc_params,
            physical_params,
            mesh,
            stats_file=args.stats_file,
            overwrite_stats_file=args.overwrite_stats_file,
            p=args.degree,
            p_penalty=p)

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


solver = BortelsSolver()

# Custom solver parameters


class CoarsenPenaltyPMGPC(P1PC):
    def coarsen_form(self, form, replace_dict):
        k = 1
        replace_dict[solver.penalty_degree] = Constant(k)
        replace_dict[solver.penalty_degreeU] = Constant(k)
        return super(CoarsenPenaltyPMGPC, self).coarsen_form(form, replace_dict)


U_is = solver.num_mass
is_list = [str(i) for i in range(solver.num_mass)]
C_is = ",".join(is_list)


if args.degree == 1:
    asm = {"pc_type": "asm",
           "pc_asm_overlap": 1,
           "sub": {
               "pc_type": "ilu",
               "pc_factor_levels": 0,
           }
           }
    gmg = {"pc_type": "mg",
           "mg_levels_ksp_type": "richardson",
           "mg_levels": asm,
           "mg_coarse_ksp_type": "preonly",
           "mg_coarse": {
               "mat_type": "aij",
               "pc_type": "telescope",
               "pc_telescope_reduction_factor": REDFACTOR,
               "pc_telescope_subcomm_type": "contiguous",
               "telescope_pc_type": "lu",
               "telescope_pc_factor_mat_solver_type": "mumps",
           }
           }

else:

    asm = {"pc_type": "python",
           "pc_python_type": "firedrake.AssembledPC",
           "assembled": {
               "pc_type": "asm",
               "pc_asm_overlap": 1,
               "sub": {
                   "pc_type": "ilu",
                   "pc_factor_levels": 0,
               },
           },
           }
    pmgasm = {"pc_type": "python",
              "mat_type": "matfree",
              "pc_python_type": __name__+".CoarsenPenaltyPMGPC",
              "pmg_mg_levels_ksp_type": "chebyshev",
              "pmg_mg_levels_ksp_max_it": 4,
              "pmg_mg_levels_ksp_monitor": None,
              "pmg_mg_levels_ksp_converged_reason": None,
              "pmg_mg_levels_ksp_norm_type": "unpreconditioned",
              "pmg_mg_levels_": {
                  "pc_type": "python",
                  "pc_python_type": __name__ + "." + "CellIntegralPC",
                  "assembled": {
                      "pc_type": "jacobi"
                  }
              },
              "pmg_coarse_mat_type": "aij",
              "pmg_mg_coarse": {
                  "ksp_type": "gmres",
                  "ksp_rtol": 1e-1,
                  "ksp_monitor": None,
                  "ksp_converged_reason": None,
                  "pc_type": "asm",
                  "pc_asm_overlap": 1,
                  "sub": {
                      "pc_type": "ilu",
                      "pc_factor_levels": 0,
                  }
              },
              }
    gmg = {"pc_type": "mg",
           "mg_levels_ksp_type": "richardson",
           "mg_levels": asm,
           "mg_coarse_ksp_type": "preonly",
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
           }
           }

if args.degree == 1:
    amg = {"pc_type": "hypre",
           "pc_hypre_boomeramg": {
               "strong_threshold": 0.7,
               "coarsen_type": "HMIS",
               "agg_nl": 3,
               "interp_type": "ext+i",
               "agg_num_paths": 5,
           },
           }
else:
    amg = {"pc_type": "hypre"}

pmg = {"pc_type": "python",
       "mat_type": "matfree",
       "pc_python_type": __name__+".CoarsenPenaltyPMGPC",
       # "pc_python_type": __name__+".Q1PC",
       "pmg_mg_levels_ksp_type": "chebyshev",
       "pmg_mg_levels_ksp_max_it": 4,
       "pmg_mg_levels_ksp_monitor": None,
       "pmg_mg_levels_ksp_norm_type": "unpreconditioned",
       "pmg_mg_levels_": {
           "pc_type": "python",
           "pc_python_type": __name__ + "." + "CellIntegralPC",
           "assembled": {
               "pc_type": "jacobi"
           }
       },
       "pmg_coarse_mat_type": "aij",
       "pmg_mg_coarse": {**{
           "ksp_type": "cg",
           "ksp_rtol": 1e-3,
           "ksp_monitor": None,
           "ksp_converged_reason": None,
       },
           **amg},
       }
if args.degree == 1:
    mat_type = "aij"
else:
    mat_type = "matfree"
custom_potential_solver = {
    "mat_type": mat_type,
    "snes_view": None,
    "snes_monitor": None,
    "snes_rtol": 1e-6,
    "ksp_monitor": None,
    "ksp_converged_reason": None,
    "ksp_type": "cg",
    "ksp_rtol": 1e-2,
    "log_view": None,
}
if args.degree > 2:
    custom_potential_solver = {** custom_potential_solver,
                               ** pmg}
    psolver = pmg
else:
    custom_potential_solver = {** custom_potential_solver,
                               ** amg}
    psolver = amg
if args.csolver == "asm":
    if args.degree > 2:
        csolver = asm
    else:
        csolver = asm
elif args.csolver == "gmg":
    csolver = gmg
elif args.csolver == "pmgasm":
    csolver = pmgasm


if args.degree > 2:
    U_field = "1"
    C0_field = "0"
else:
    U_field = "0"
    C0_field = "1"

solver.init_solver_parameters(
    custom_solver={
        "mat_type": mat_type,
        "snes_view": None,
        "snes_converged_reason": None,
        "snes_monitor": None,
        "snes_rtol": 1e-6,
        "ksp_monitor": None,
        "ksp_converged_reason": None,
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-3,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_"+U_field+"_fields": U_is,
        "pc_fieldsplit_"+C0_field+"_fields": 0,
        "fieldsplit_"+U_field: {**{
            "ksp_converged_reason": None,
            "ksp_rtol": 1e-1,
            "ksp_type": "cg",
        }, **psolver},
        "fieldsplit_"+C0_field: {**{
            "ksp_converged_reason": None,
            "ksp_rtol": 1e-1,
            "ksp_type": "gmres",
        }, **csolver},
    },
    custom_potential_solver=custom_potential_solver)

solver.save_solutions = False
solver.setup_solver(initial_solve=True)
solver.solve()
c1, U = solver.u.subfunctions
errc1 = errornorm(solver.C1ex, c1)
errU = errornorm(solver.Uex, U)
err = errc1 + errU
PETSc.Sys.Print("cells = {} L2err = {:.5E} C1err = {:.5E} Uerr = {:.5E}".format(
    solver.comm.allreduce(solver.mesh.cell_set.size, op=MPI.SUM), err, errc1, errU))
