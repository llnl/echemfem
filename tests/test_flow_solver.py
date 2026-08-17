from firedrake import *
from echemfem import NavierStokesFlowSolver, NavierStokesBrinkmanFlowSolver


def test_navier_stokes_poiseuille_pressure_traction():
    mesh = UnitSquareMesh(8, 8, quadrilateral=True)
    x, y = SpatialCoordinate(mesh)

    rho = 1.0
    mu = 1.0
    p_in = 1.0
    p_out = 0.0
    delta_p = p_in - p_out

    u_ex = as_vector((0.5 * delta_p / mu * y * (1 - y), 0.0))
    p_ex = p_in - delta_p * x

    flow_params = {"density": rho,
                   "dynamic viscosity": mu,
                   "inlet pressure": p_in,
                   "outlet pressure": p_out}
    boundary_markers = {"inlet pressure": (1,),
                        "outlet pressure": (2,),
                        "no slip": (3, 4)}

    solver = NavierStokesFlowSolver(mesh, flow_params, boundary_markers)
    solver.setup_solver()
    solver.solver.solve()
    u, p = solver.soln.subfunctions

    assert errornorm(u_ex, u, norm_type="L2") < 1e-8
    assert errornorm(p_ex, p, norm_type="L2") < 1e-8


def test_navier_stokes_couette_velocity_dirichlet():
    mesh = UnitSquareMesh(8, 8, quadrilateral=True)
    _, y = SpatialCoordinate(mesh)

    top_velocity = 0.25
    u_ex = as_vector((top_velocity * y, 0.0))

    flow_params = {"density": 1.0,
                   "dynamic viscosity": 1.0,
                   "inlet velocity": u_ex}
    boundary_markers = {"no slip": (3,),
                        "inlet velocity": (1, 2, 4)}

    solver = NavierStokesFlowSolver(mesh, flow_params, boundary_markers)
    solver.setup_solver()
    solver.solver.solve()
    u, p = solver.soln.subfunctions

    assert errornorm(u_ex, u, norm_type="L2") < 1e-8
    assert norm(p - assemble(p * dx), norm_type="L2") < 1e-8


def test_navier_stokes_brinkman_pressure_driven_channel():
    mesh = UnitSquareMesh(16, 16, quadrilateral=True)
    x, y = SpatialCoordinate(mesh)

    rho = 1.0
    nu = 1.0
    permeability = 0.1
    p_in = 1.0
    p_out = 0.25
    delta_p = p_in - p_out

    alpha = sqrt(nu / permeability)
    u_scale = delta_p * permeability / (rho * nu)
    u_ex = as_vector((
        u_scale * (1 - cosh(alpha * (y - 0.5)) / cosh(0.5 * alpha)),
        0.0))
    p_ex = p_in - delta_p * x

    flow_params = {"density": rho,
                   "kinematic viscosity": nu,
                   "permeability": permeability,
                   "inlet pressure": p_in,
                   "outlet pressure": p_out}
    boundary_markers = {"inlet pressure": (1,),
                        "outlet pressure": (2,),
                        "no slip": (3, 4)}

    solver = NavierStokesBrinkmanFlowSolver(mesh, flow_params, boundary_markers)
    solver.setup_solver()
    solver.solver.solve()
    u, p = solver.soln.subfunctions

    assert errornorm(u_ex, u, norm_type="L2") < 1e-3
    assert errornorm(p_ex, p, norm_type="L2") < 1e-3
