from abc import ABC
import os
from FIAT import quadrature
from firedrake import *
from petsc4py import PETSc
from mpi4py import MPI
from pyop2.mpi import COMM_WORLD
import pandas
from echemfem import CylindricalMeasure

import FIAT
import finat


def gauss_lobatto_legendre_line_rule(degree):
    fiat_make_rule = FIAT.quadrature.GaussLobattoLegendreQuadratureLineRule
    fiat_rule = fiat_make_rule(FIAT.ufc_simplex(1), degree + 1)
    finat_ps = finat.point_set.GaussLobattoLegendrePointSet
    points = finat_ps(fiat_rule.get_points())
    weights = fiat_rule.get_weights()
    return finat.quadrature.QuadratureRule(points, weights)


def gauss_lobatto_legendre_cube_rule(dimension, degree):
    make_tensor_rule = finat.quadrature.TensorProductQuadratureRule
    result = gauss_lobatto_legendre_line_rule(degree)
    for _ in range(1, dimension):
        line_rule = gauss_lobatto_legendre_line_rule(degree)
        result = make_tensor_rule([result, line_rule])
    return result


class EchemSolver(ABC):
    """Base class for an electrochemical model solver.

    This class is used to create solvers for the transport of multiple ion (or
    non-charged species).

    Attributes:
        conc_params (list of dict): List containing one dictionary for each species.
            Each dictionary contains physical parameters for the species.
        physical_params (dict): Dictionary containing physical parameters 
        mesh (:class:`firedrake.mesh.MeshGeometry`): Mesh object from firedrake
        echem_params (list, optional): List containing one dictionary for each charge-transfer reaction.
        gas_params (list, optional): List containing one dictionary for each gaseous 
            species. Each dictionary contains physical parameters for the gaseous
            species. Note that this implementation is not yet validated.
        stats_file (str, optional): File name for performance statistics
        overwrite_stats_file (bool, optional): Set to True to overwrite new file.
        p (int, optional): Polynomial degree of finite elements.
        family (str, optional): Finite element family. Choose between "CG" and "DG".
        p_penalty (str, optional): Polynomial degree of the DG penalization. Only used for P-multigrid.,
        SUPG (bool, optional): Streamline upwind diffusion for stablization of the advection-migration term. Only used for CG.
        cylindrical (bool, optional): if True, uses symmetric cylindrical coordinates (r, z).
    """

    def __init__(
            self,
            conc_params,
            physical_params,
            mesh,
            echem_params=[],
            homog_params=[],
            gas_params=[],
            stats_file="",
            overwrite_stats_file=False,
            p=1,
            family="DG",
            p_penalty=None,
            SUPG=False,
            cylindrical=False):

        self.physical_params = physical_params
        self.conc_params = conc_params
        self.echem_params = echem_params
        self.homog_params = homog_params
        self.gas_params = gas_params
        self.num_c = len(conc_params)
        self.num_g = len(gas_params)

        self.stats_file = stats_file
        self.overwrite_stats_file = overwrite_stats_file

        self.comm = COMM_WORLD

        self.mesh = mesh
        self.set_boundary_markers()
        for mrk in self.boundary_markers:
            if isinstance(self.boundary_markers[mrk], int):
                PETSc.Sys.Print("*** WARNING: boundary marker converted from int to tuple")
                self.boundary_markers[mrk] = (self.boundary_markers[mrk],)

        self.flow = {"advection": False,
                     "diffusion": False,
                     "migration": False,
                     "finite size": False,
                     "electroneutrality": False,  # elimination and charge eqn
                     "electroneutrality full": False,  # no elimination
                     "poisson": False,
                     "porous": False,
                     "darcy": False,  # adds pressure variable(s), velocity(ies) defined by Darcy's law.
                     "advection gas only": False,  # Need a second velocity for gas phase
                     "diffusion finite size_BK": False,
                     "diffusion finite size_CS": False,
                     "diffusion finite size_SP": False,
                     "diffusion finite size_SMPNP": False,
                     "diffusion finite size_BMCSL": False
                     }

        self.SUPG = SUPG
        self.save_solutions = True  # hidden option

        for physics in physical_params["flow"]:
            self.flow[physics] = True
        if (not self.flow["migration"]) and self.flow["electroneutrality"]:
            raise NotImplementedError(
                'Electroneutrality constraint requires Migration')
        if self.flow["migration"] and (
            (not self.flow["electroneutrality"]) and (
                not self.flow["poisson"]) and (not self.flow["electroneutrality full"])):
            raise NotImplementedError(
                'Migration requires Electroneutrality or Poisson')
        if self.flow["poisson"] and (self.flow["electroneutrality"]
                                     or self.flow["electroneutrality full"]):
            raise NotImplementedError(
                'Electroneutrality not compatible with Poisson')
        for physics in physical_params["flow"]:
            if "finite size" in physics and family == "DG":
                raise NotImplementedError(
                    'finite size effects only implemented with CG')
        if self.flow["darcy"] and (not self.flow["porous"]):
            raise NotImplementedError('Darcy requires a porous model')
        if self.gas_params and not self.flow["darcy"]:
            raise NotImplementedError(
                'Two-phase currently only supported for Darcy flow')
        if self.flow["darcy"] and not (
                self.flow["advection"] or self.flow["advection gas only"]):
            raise NotImplementedError(
                'Darcy requires advection or advection gas only')
        if not self.gas_params and self.flow["advection gas only"]:
            raise NotImplementedError(
                'advection gas only requires gaseous species')

        # list of conc_params indices that are not eliminated
        self.idx_c = list(range(self.num_c))
        # Other parts of the code currently assumes the last concentration is
        # the eliminated one
        if self.flow["electroneutrality"]:
            i = 0
            for idx, c in enumerate(conc_params):
                if c.get("eliminated") is not None:
                    i += 1
                    self.i_el = idx
            # By default the last conc_params is eliminated
            if i == 0:
                conc_params[-1]["eliminated"] = True
                self.i_el = idx
            self.idx_c.remove(self.i_el)
            print = PETSc.Sys.Print
            print("Eliminating species: " + conc_params[self.i_el]["name"])
            assert not conc_params[self.i_el]["z"] == 0, "Eliminated species must have non-zero charge. Add \"eliminated\": True to a species with non-zero charge."
            assert i < 2, "Can only eliminate one aqueous concentration"

        if self.gas_params:
            i = 0
            for g in gas_params:
                if g.get("eliminated") is not None:
                    i += 1
            if i == 0:
                gas_params[-1]["eliminated"] = True
            assert i < 2, "Can only eliminate one gaseous concentration"

        # number of primary varibles
        self.num_liquid = self.num_c
        self.num_gas = self.num_g
        if self.flow["electroneutrality"]:
            self.num_liquid -= 1
        if self.gas_params:
            self.num_gas -= 1
        self.num_mass = self.num_liquid + self.num_gas

        # Setup spaces
        self.poly_degree = p
        self.poly_degreeU = self.poly_degree
        self.poly_degreep = self.poly_degree
        self.family = family
        # DG penalization parameter
        self.C_IP = Constant(10.)
        # These DG parameters are defined here for PMG
        if p_penalty is None:
            p_penalty = Constant(p)
        self.penalty_degree = p_penalty  # Constant(self.poly_degree)
        self.penalty_degreeU = p_penalty  # Constant(self.poly_degreeU)
        self.penalty_degreep = p_penalty  # Constant(self.poly_degreep)

        quadrature_rule = None
        quadrature_rule_face = None
        if cylindrical:
            print = PETSc.Sys.Print
            print("Using cylindrical coordinates")
            # cylindrical coordinates with azimuthal symmetry (2D mesh)
            if mesh.topological_dimension() != 2:
                raise NotImplementedError(
                    'Cylindrical coordinates require a 2D mesh')
            r, z = SpatialCoordinate(mesh)
            _dx = CylindricalMeasure(r, 'cell')
            _ds = CylindricalMeasure(r, 'exterior_facet')
            _dS = CylindricalMeasure(r, 'interior_facet')
        else:
            _dx = dx
            _ds = ds
            _dS = dS

        def _internal_dx(subdomain_id=None, domain=None):
            return _dx(
                subdomain_id=subdomain_id,
                domain=domain,
                scheme=quadrature_rule,
                degree=p+1)

        if mesh.layers:
            # This doesn't include top and bottom surfaces
            def _internal_ds(subdomain_id=None, domain=None):
                return ds_v(
                    subdomain_id=subdomain_id,
                    domain=domain,
                    scheme=quadrature_rule_face,
                    degree=p+1)

            def _internal_dS(subdomain_id=None, domain=None):
                return dS_v(subdomain_id=subdomain_id,
                            domain=domain,
                            scheme=quadrature_rule_face,
                            degree=p+1) + dS_h(subdomain_id=subdomain_id,
                                                                domain=domain,
                                                                scheme=quadrature_rule_face,
                                                                degree=p+1)
        else:
            def _internal_ds(subdomain_id=None, domain=None):
                return _ds(
                    subdomain_id=subdomain_id,
                    domain=domain,
                    scheme=quadrature_rule_face,
                    degree=p+1)

            def _internal_dS(subdomain_id=None, domain=None):
                return _dS(
                    subdomain_id=subdomain_id,
                    domain=domain,
                    scheme=quadrature_rule_face,
                    degree=p+1)

        self.ds = _internal_ds
        self.dx = _internal_dx
        self.dS = _internal_dS

        # Set base function spaces
        self.set_function_spaces(mesh.layers, family)

        # MixedFunctionSpace - unknownwise ordering
        spaces = []
        for i in range(self.num_mass):
            spaces.append(self.V)
        if self.flow["poisson"] or self.flow["electroneutrality"] or self.flow["electroneutrality full"]:
            spaces.append(self.Vu)
            if self.flow["porous"]:
                spaces.append(self.Vu)
        if self.flow["darcy"]:
            spaces.append(self.Vp)  # one pressure
            if self.gas_params and self.flow["advection"]:  # two pressures
                spaces.append(self.Vp)
        self.W = MixedFunctionSpace(spaces)

        # VectorFunctionSpace - pointwise ordering
        self.vector = False
        if self.vector:
            num_space = self.num_mass
            if self.flow["migration"]:
                num_space += 1
                if self.flow["porous"]:
                    num_space += 1
            if self.flow["darcy"]:
                num_space += 1
                if self.gas_params and self.flow["advection"]:
                    num_space += 1
            self.W = VectorFunctionSpace(mesh, family, self.poly_degree, dim=num_space)

        # pointwise ordering for concentrations and mass fractions
        self.vector_mix = False
        spaces = []
        if self.vector_mix:
            self.Vc = VectorFunctionSpace(
                mesh, family, self.poly_degree, dim=self.num_mass)
            spaces.append(self.Vc)
            if self.flow["poisson"] or self.flow["electroneutrality"] or self.flow["electroneutrality full"]:
                spaces.append(self.Vu)
                if self.flow["porous"]:
                    spaces.append(self.Vu)
            if self.flow["darcy"]:
                spaces.append(self.Vp)
                if self.gas_params and self.flow["advection"]:
                    spaces.append(self.Vp)
            self.W = MixedFunctionSpace(spaces)

        # setup solution and test functions
        u = Function(self.W, name="solution")
        v = TestFunctions(self.W)
        us = split(u)
        if self.vector_mix:
            us = split(us[0]) + us[1:]
            v = split(v[0]) + v[1:]
        # solutions and equations are in the following order:
        # n_liquid aqueous concentrations, aqueous mass conservation equation
        # n_gas gaseous concentrations, gaseous mass conservation equation
        # Liquid potential, charge conservation equation / liquid Poisson equation
        # Solid potential, Poisson equation
        # Liquid pressure, liquid pressure equation
        # Gas pressure, gas pressure equation

        # indices of the different variables in the solution vector. only for u[i] if using vector_mix
        self.i_c = {}
        for i in range(self.num_liquid):
            self.i_c.update({conc_params[self.idx_c[i]]["name"]: i})
        self.i_g = {}
        for i in range(self.num_gas):
            self.i_g.update({gas_params[i]["name"]: self.num_liquid + 1})
        if self.flow["electroneutrality"] or self.flow["poisson"] or self.flow["electroneutrality full"]:
            self.i_Ul = self.num_mass
            if self.flow["porous"]:
                self.i_Us = self.num_mass + 1

        # setup pressure variables for Darcy's law
        if self.flow["darcy"]:
            if self.flow["poisson"] or self.flow["electroneutrality"] or self.flow["electroneutrality full"]:
                if self.flow["advection"]:
                    self.i_pl = self.num_mass + 2
                    self.pl = us[self.i_pl]
                    if self.gas_params:
                        self.i_pg = self.num_mass + 3
                        self.pg = us[self.i_pg]
                else:  # advection gas only
                    self.i_pg = self.num_mass + 2
                    self.pg = us[self.i_pg]
            else:
                if self.flow["advection"]:
                    self.i_pl = self.num_mass
                    self.pl = us[self.i_pl]
                    if self.gas_params:
                        self.i_pg = self.num_mass + 1
                        self.pg = us[self.i_pg]
                else:  # advection gas only
                    self.i_pg = self.num_mass
                    self.pg = us[self.i_pg]
        if self.flow["advection gas only"]:
            pl = self.physical_params["p_gas"]
            self.i_pl = self.i_pg  # for solver parameters
            if isinstance(pl, (float, int)):
                self.pl = Constant(pl)
            else:
                self.pl = Function(self.Vp).interpolate(pl)

        if self.gas_params:
            self.Sw = self.saturation(self.pl, self.pg)
        else:
            self.Sw = self.physical_params.get("saturation")
            if self.Sw is None:
                self.Sw = Constant(1.0)
        if self.gas_params:
            self.set_gas_density(
                us[self.num_liquid:self.num_liquid + self.num_gas], self.pg, gas_params)

        # setup applied voltage
        if physical_params.get("U_app") is not None:
            U_app = physical_params["U_app"]
            U_app_ufl = physical_params.get("U_app_ufl") # keep UFL expression
            if isinstance(U_app, (Constant, Function)) or U_app_ufl:
                self.U_app = U_app
            elif isinstance(U_app, (float, int)):
                self.U_app = Constant(U_app)
            else:
                self.U_app = Function(self.Vu).interpolate(U_app)

        if self.flow["darcy"]:
            self.set_darcy_velocity()
        elif self.flow["advection"]:
            self.set_velocity()

        if self.homog_params:
            self.setup_bulk_reactions()

        self.setup_forms(us, v)

        self.u = u

        self.init_solver_parameters()

    def setup_forms(self, us, v):
        """ Setup weak forms
        """
        self.Form, self.bcs = self.steady_forms(us, v)

    def steady_forms(self, us, v):
        """ Add steady-state portion of equations to weak forms
        """
        conc_params = self.conc_params
        gas_params = self.gas_params

        Form = 0.0
        bcs = []  # Dirichlet BCs for CG
        # mass conservation of aqueous species
        for i in range(self.num_liquid):
            if self.flow["electroneutrality full"] and i == self.num_liquid-1:
                test_fn = v[i+1]  # to avoid zeroes on the diagonal. last species must have a charge
            else:
                test_fn = v[i]
            conc_params[self.idx_c[i]]["i_c"] = i
            a, bc = self.mass_conservation_form(
                us[i], test_fn, conc_params[self.idx_c[i]], u=us)
            weight = conc_params[self.idx_c[i]].get("residual weight")
            if weight is None:
                weight = 1.0
            Form += weight * a
            bcs += bc

        # mass conservation of gaseous species
        for i in range(self.num_gas):
            if gas_params[i].get("eliminated") is None:
                j = i + self.num_liquid
                gas_params[i]["i_g"] = j
                a, bc = self.gas_mass_conservation_form(
                    us[j], v[j], gas_params[i], u=us)
                Form += a
                bcs += bc
        # add gas source terms to gas mass conservation equations: for testing
        # only
        Form += self.gas_sources(us, v)
        # add dissolution terms to all equations
        if gas_params:
            Form += self.dissolutions(us, v, conc_params, gas_params)

        # TODO: dissolutions for eliminated concentrations
        # charge conservation
        if self.flow["electroneutrality"]:
            a, bc = self.charge_conservation_form(us,
                                                  v[self.num_mass], conc_params)
            Form += a
            bcs += bc
        # or Poisson equation
        elif self.flow["poisson"]:
            a, bc = self.potential_poisson_form(us,
                                                v[self.num_mass],
                                                conc_params,
                                                solid=False)
            Form += a
            bcs += bc
        elif self.flow["electroneutrality full"]:
            a, bc = self.electroneutrality_form(us, v[self.num_liquid - 1], conc_params)
            weight = self.physical_params.get("electroneutrality residual weight")
            if weight is None:
                weight = 1.0
            Form += weight * a
            bcs += bc
        # Poisson equation for the solid potential
        if self.flow["porous"] and (
                self.flow["poisson"] or self.flow["electroneutrality"] or self.flow["electroneutrality full"]):
            a, bc = self.potential_poisson_form(us,
                                                v[self.num_mass + 1],
                                                conc_params,
                                                solid=True)
            weight = self.physical_params.get("solid potential residual weight")
            if weight is None:
                weight = 1.0
            Form += weight * a
            bcs += bc
        # Single phase Darcy
        if self.flow["darcy"] and self.flow["advection"]:
            a, bc = self.liquid_pressure_form(self.pl,
                                              v[self.i_pl], conc_params, u=us)
            Form += a
            bcs += bc
        # Two-phase Darcy
        if self.flow["darcy"] and self.gas_params:
            # Form += self.gas_pressure_form(self.pg, v[self.i_pg], gas_params, u=us)
            a, bc = self.gas_mass_conservation_form_pressure(
                self.pg, v[self.i_pg], gas_params[self.num_gas], u=us)
            Form += a
            bcs += bc

        return Form, bcs

    def print_solver_info(self):
        print = PETSc.Sys.Print

        print("*** ECHEM")

        print("Activated flow components")
        for component, active in self.flow.items():
            if active:
                if component == "advection" or component == "migration":
                    poly_degree = self.poly_degree
                elif component == "diffusion" or component == "electroneutrality":
                    poly_degree = self.poly_degreeU

                print("  {} \t order = {}".format(component, poly_degree))

        print("Species information")
        for c in self.conc_params:
            print("> {}".format(c["name"]))
            if self.flow["electroneutrality"] or (self.flow["poisson"] and self.flow["migration"]) or self.flow["electroneutrality full"]:
                print("  z = {}".format(c["z"]))

        print("Solver information")
        total_dof_count = 0
        print("variable \t #dofs")
        for c in self.conc_params:
            print("{} \t\t {}".format(c["name"], self.V.dim()))
            total_dof_count += self.V.dim()

        if "poisson" in self.physical_params["flow"] or "electroneutrality" in self.physical_params["flow"] or self.flow["electroneutrality full"]:
            print("Potential \t {}".format(self.Vu.dim()))
            total_dof_count += self.Vu.dim()

        print("Total \t\t {}".format(total_dof_count))

        print("#ranks \t\t {}".format(MPI.Comm.Get_size(MPI.COMM_WORLD)))
        print("#dofs/rank \t {}".format(int(total_dof_count
              / MPI.Comm.Get_size(MPI.COMM_WORLD))))

    def setup_solver(self, initial_guess=True, initial_solve=True):
        """Sets up the initial guess and solver

        This creates :class:`firedrake.variational_solver.NonlinearVariationalProblem` and
        :class:`firedrake.variational_solver.NonlinearVariationalSolver` objects stored in
        self.problem and self.echem_solver, respectively.

        Args:
            initial_guess (bool): If True, sets all concentrations to their
                given "bulk" value.
            initial_solve (bool): If True, solves the system assuming
                constant concentrations, which provides good initial
                guesses for the potential(s)
        """
        self.print_solver_info()

        u = self.u
        us = split(u)
        if self.vector_mix:
            us = split(us[0]) + us[1:]
        Form = self.Form
        if initial_guess:
            if self.vector_mix:
                i_Ul = 0
                i_Us = 0
                if self.flow["electroneutrality"] or self.flow["poisson"] or self.flow["electroneutrality full"]:
                    i_Ul = 1
                    i_Us = 1
                    if self.flow["porous"]:
                        i_Us = 2
                if self.flow["darcy"]:
                    i_pl = i_Us + 1
                    if self.gas_params:
                        i_pg = i_pl + 1
            else:
                if self.flow["electroneutrality"] or self.flow["poisson"] or self.flow["electroneutrality full"]:
                    i_Ul = self.i_Ul
                    if self.flow["porous"]:
                        i_Us = self.i_Us
                if self.flow["darcy"]:
                    i_pl = self.i_pl
                    if self.gas_params:
                        i_pg = self.i_pg
            for i in range(self.num_liquid):
                bulk = self.conc_params[self.idx_c[i]]["bulk"]
                if isinstance(bulk, (float, int)):
                    if self.vector_mix:
                        u.sub(0).sub(i).assign(bulk)
                    else:
                        u.sub(i).assign(bulk)
                else:
                    if self.vector_mix:
                        u.sub(0).sub(i).interpolate(bulk)
                    else:
                        u.sub(i).interpolate(bulk)
            for i in range(self.num_gas):
                gas = self.gas_params[i]["gas"]
                if isinstance(gas, (float, int)):
                    if self.vector_mix:
                        u.sub(0).sub(i + self.num_liquid).assign(gas)
                    else:
                        u.sub(i + self.num_liquid).assign(gas)
                else:
                    if self.vector_mix:
                        u.sub(0).sub(i + self.num_liquid).interpolate(gas)
                    else:
                        u.sub(i + self.num_liquid).interpolate(gas)

            if self.flow["darcy"]:
                p_gas = self.physical_params["p_gas"]
                if self.flow["advection"]:
                    if isinstance(p_gas, (float, int)):
                        u.sub(i_pl).assign(p_gas)
                    else:
                        u.sub(i_pl).interpolate(p_gas)
                if self.gas_params:
                    if isinstance(p_gas, (float, int)):
                        u.sub(i_pg).assign(p_gas)
                    else:
                        u.sub(i_pg).interpolate(p_gas)

            if self.flow["porous"] and (
                    self.flow["electroneutrality"] or self.flow["poisson"] or self.flow["electroneutrality full"]):
                u.sub(i_Ul).assign(Constant(0))
                try:
                    u.sub(i_Us).assign(self.U_app)
                except:
                    u.sub(i_Us).interpolate(self.U_app)
                if initial_solve:
                    Wu = self.Vu * self.Vu
                    U0 = Function(Wu)
                    U0s = split(U0)
                    v0, v1 = TestFunctions(Wu)
                    u0 = [us[i] for i in range(self.num_mass)]
                    u0.append(U0s[0])
                    u0.append(U0s[1])
                    if self.flow["electroneutrality"]:
                        a0, bcs = self.charge_conservation_form(
                            u0, v0, self.conc_params, W=Wu, i_bc=0)
                    else:
                        a0, bcs = self.potential_poisson_form(
                            u0, v0, self.conc_params, W=Wu, i_bc=0)
                    a, bc = self.potential_poisson_form(u0, v1,
                                                        self.conc_params, solid=True, W=Wu, i_bc=1)
                    a0 += a
                    bcs += bc
                    solve(
                        a0 == 0,
                        U0,
                        bcs=bcs,
                        solver_parameters=self.potential_solver_parameters,
                        options_prefix="echem_potential_initial_guess_")
                    Ul, Us = U0.subfunctions
                    VTKFile("results/liquid_potential0.pvd").write(Function(self.Vu,
                                                                         name="Liquid Potential").assign(Ul))
                    VTKFile("results/solid_potential0.pvd").write(Function(self.Vu,
                                                                        name="Solid Potential").assign(Us))
                    u.sub(i_Ul).assign(U0.sub(0))
                    u.sub(i_Us).assign(U0.sub(1))

            elif (self.flow["electroneutrality"] or self.flow["poisson"] or self.flow["electroneutrality full"]):
                U0 = Function(self.Vu)
                if hasattr(self, "U_app"):
                    try:
                        U0.assign(self.U_app / 2)
                    except:
                        U0.interpolate(self.U_app / 2)

                if initial_solve:
                    v0 = TestFunction(self.Vu)
                    u0 = [us[i] for i in range(self.num_mass)]
                    u0.append(U0)
                    if self.flow["electroneutrality"]:
                        a0, bcs = self.charge_conservation_form(u0, v0, self.conc_params, W=self.Vu, i_bc=0)
                    else:
                        a0, bcs = self.potential_poisson_form(u0, v0, self.conc_params, W=self.Vu, i_bc=0)
                    solve(
                        a0 == 0,
                        U0,
                        bcs=bcs,
                        solver_parameters=self.potential_solver_parameters,
                        options_prefix="echem_potential_initial_guess")
                u.sub(i_Ul).assign(U0)

            if self.flow["darcy"] and self.gas_params and self.flow["advection"]:
                Wu = self.Vp * self.Vp
                p0 = Function(Wu)
                p_gas = self.physical_params["p_gas"]
                if isinstance(p_gas, (float, int)):
                    p0.sub(0).assign(p_gas)
                    p0.sub(1).assign(p_gas)
                else:
                    p0.sub(0).interpolate(p_gas)
                    p0.sub(1).interpolate(p_gas)
                if initial_solve:
                    p0s = split(p0)
                    v0, v1 = TestFunctions(Wu)
                    u0 = [us[i] for i in range(self.num_mass + 2)]
                    u0.append(p0s[0])
                    u0.append(p0s[1])
                    a0, bcs = self.liquid_pressure_form(
                        p0s[0], v0, self.conc_params, u=u0, W=Wu, i_bc=0)
                    a, bc = self.gas_mass_conservation_form_pressure(
                        p0s[1], v1, self.gas_params[self.num_gas], u=u0, W=Wu, i_bc=1)
                    a0 += a
                    bcs += bc
                    pressure_solver_parameters = {"snes_type": "newtonls",
                                                  "snes_linesearch_type": "l2",
                                                  "snes_rtol": 1e-8,
                                                  "snes_max_it": 20,
                                                  "mat_type": "aij",
                                                  "ksp_type": "preonly",
                                                  "pc_type": "lu",
                                                  }
                    solve(
                        a0 == 0,
                        p0,
                        bcs=bcs,
                        solver_parameters=pressure_solver_parameters)
                    pl, pg = p0.subfunctions
                    VTKFile("results/liquid_pressure0.pvd").write(Function(self.Vp,
                                                                        name="Liquid Pressure").assign(pl))
                    VTKFile("results/gas_pressure0.pvd").write(Function(self.Vp,
                                                                     name="Gas Pressure").assign(pg))
                u.sub(i_pl).assign(p0.sub(0))
                u.sub(i_pg).assign(p0.sub(1))

            if self.flow["advection gas only"]:
                if isinstance(p_gas, (float, int)):
                    u.sub(i_pg).assign(p_gas)
                else:
                    u.sub(i_pg).interpolate(p_gas)

        self.echem_problem = NonlinearVariationalProblem(Form, u, bcs=self.bcs)
        self.echem_solver = NonlinearVariationalSolver(
            self.echem_problem,
            solver_parameters=self.solver_parameters,
            options_prefix="echem_system")
        self.echem_solver.snes.setConvergenceHistory()
        self.echem_solver.snes.ksp.setConvergenceHistory()

    def solve(self):
        """Solves the problem and outputs the solutions
        """
        self.echem_solver.solve()
        if self.save_solutions:
            self.output_state(self.u)

        if len(self.stats_file) > 0:
            solver = self.echem_solver
            comm = self.comm

            snes = PETSc.Log.Event("SNESSolve").getPerfInfo()
            ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()
            pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
            pcapply = PETSc.Log.Event("PCApply").getPerfInfo()
            jac = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()
            residual = PETSc.Log.Event("SNESFunctionEval").getPerfInfo()

            snes_time = comm.allreduce(snes["time"], op=MPI.SUM) / comm.size
            jac_time = comm.allreduce(jac["time"], op=MPI.SUM) / comm.size
            residual_time = comm.allreduce(
                residual["time"], op=MPI.SUM) / comm.size
            ksp_time = comm.allreduce(ksp["time"], op=MPI.SUM) / comm.size
            pcsetup_time = comm.allreduce(
                pcsetup["time"], op=MPI.SUM) / comm.size
            pcapply_time = comm.allreduce(
                pcapply["time"], op=MPI.SUM) / comm.size

            newton_its = solver.snes.getIterationNumber()
            ksp_its = solver.snes.getLinearSolveIterations()
            num_cells = comm.allreduce(self.mesh.cell_set.size, op=MPI.SUM)

            stats = os.path.abspath(self.stats_file)

            if COMM_WORLD.rank == 0:
                snes_history, linear_its = solver.snes.getConvergenceHistory()
                ksp_history = solver.snes.ksp.getConvergenceHistory()

                # some hard-coding for the DG paper
                data = {
                    "num_processes": comm.size,
                    "num_cells": num_cells,
                    "dimension": 3,
                    "degreeC": self.poly_degree,
                    "degreeU": self.poly_degreeU,
                    "dofs": self.u.dof_dset.layout_vec.getSize(),
                    "name": "bortels_threeion_extruded_3d_nondim",
                    "snes_its": newton_its,
                    "ksp_its": ksp_its,
                    "SNESSolve": snes_time,
                    "KSPSolve": ksp_time,
                    "PCSetUp": pcsetup_time,
                    "PCApply": pcapply_time,
                    "JacobianEval": jac_time,
                    "FunctionEval": residual_time,
                    "mesh_name": self.mesh.name,
                    "csolver": self.csolver,
                }

                if not os.path.exists(os.path.dirname(stats)):
                    os.makedirs(os.path.dirname(stats))

                if self.overwrite_stats_file:
                    mode = "w"
                    header = True
                else:
                    mode = "a"
                    header = not os.path.exists(stats)

                df = pandas.DataFrame(data, index=[0])
                df.to_csv(stats, index=False, mode=mode, header=header)

    def output_state(self, u, prefix="results/", initiate=True, **kwargs):
        """ Outputs the provided state variables in a pvd file

            Args:
                u (:class:`firedrake.function.Function`): State to output. Must
                    be same FunctionSpace as self.u
                prefix (str): Path to results directory
                initiate (bool): if True, create the output file
                **kwargs: Arbitrary keyword arguments passed to VTKFile.write

        """
        PETSc.Sys.Print("Writing solutions. This may take a while...")
        if self.vector:
            uviz = u
        elif self.vector_mix:
            uvi = u.subfunctions
            uviz = ()
            for i in range(self.num_mass):
                uviz += (uvi[0][i],)
            uviz += uvi[1:]
        else:
            uviz = u.subfunctions

        if initiate:
            self.output_file = VTKFile(prefix + "collection.pvd")
        r = self.output_file
        collection = []

        if self.flow["advection"]:
            vel_vizfs = VectorFunctionSpace(self.mesh, self.family, 1)
            projected_velocity = project(self.vel, vel_vizfs)
            projected_velocity.rename("velocity")
            collection.append(projected_velocity)

        for i in range(self.num_liquid):
            collection.append(
                Function(
                    self.V,
                    name=self.conc_params[self.idx_c[i]]["name"]).assign(
                    uviz[i]))

        if self.flow["poisson"] or self.flow["electroneutrality"] or self.flow["electroneutrality full"]:
            collection.append(Function(self.Vu,
                              name="Liquid Potential").assign(uviz[self.num_mass]))
            if self.flow["porous"]:
                collection.append(Function(self.Vu,
                                  name="Solid Potential").assign(uviz[self.i_Us]))

        if self.flow["electroneutrality"]:
            C_el = 0.0
            for i in range(self.num_liquid+1):
                if i == self.i_el:
                    z_el = self.conc_params[i]["z"]
                    C_ND_el = self.conc_params[i].get("C_ND")
                    if C_ND_el is None:
                        C_ND_el = 1.0
                else:
                    C_ND = self.conc_params[i].get("C_ND")
                    if C_ND is None:
                        C_ND = 1.0
                    C_el += self.conc_params[i]["z"] * uviz[i] * C_ND
            C_el = -C_el / z_el / C_ND_el
            collection.append(
                Function(self.V, name=self.conc_params[self.i_el]["name"]).assign(C_el))

        r.write(*collection, **kwargs)

    def write_results_old(self):
        # Output solutions
        if self.vector:
            uviz = u
        elif self.vector_mix:
            uvi = u.subfunctions
            uviz = ()
            for i in range(self.num_mass):
                uviz += (uvi[0][i],)
            uviz += uvi[1:]
        else:
            uviz = u.subfunctions

        # Visualize velocity for convenience
        vel_vizfs = VectorFunctionSpace(self.mesh, family, 1)
        projected_velocity = project(self.vel, vel_vizfs)
        projected_velocity.rename("velocity")
        VTKFile("results/" + "velocity" + ".pvd").write(projected_velocity)

        for i in range(self.num_liquid):
            VTKFile(
                "results/"
                + self.conc_params[i]["name"]
                + ".pvd").write(
                Function(
                    self.V,
                    name=self.conc_params[i]["name"]).assign(
                    uviz[i]))
        X_el = 1.0
        for i in range(self.num_gas):
            j = i + self.num_liquid
            VTKFile(
                "results/"
                + self.gas_params[i]["name"]
                + ".pvd").write(
                Function(
                    self.V,
                    name=self.gas_params[i]["name"]).assign(
                    uviz[j]))
            X_el -= uviz[j]
        if self.gas_params:
            VTKFile("results/" + self.gas_params[self.num_gas]["name"] + ".pvd").write(
                Function(self.V, name=self.gas_params[self.num_gas]["name"]).assign(X_el))
            VTKFile("results/saturation.pvd").write(
                Function(self.V, name="Saturation").interpolate(self.Sw))
        if self.flow["poisson"] or self.flow["electroneutrality"] or self.flow["electroneutrality full"]:
            VTKFile("results/liquid_potential.pvd").write(Function(self.Vu,
                                                                name="Liquid Potential").assign(uviz[self.num_mass]))
            if self.flow["porous"]:
                VTKFile("results/solid_potential.pvd").write(Function(self.Vu,
                                                                   name="Solid Potential").assign(uviz[self.num_mass + 1]))
        if self.flow["electroneutrality"]:
            C_el = 0.0
            for i in range(self.num_liquid):
                C_el += self.conc_params[i]["z"] * uviz[i]
            C_el = -C_el / self.conc_params[self.num_liquid]["z"]
            VTKFile("results/" + self.conc_params[self.num_liquid]["name"] + ".pvd").write(
                Function(self.V, name=self.conc_params[self.num_liquid]["name"]).assign(C_el))
        if self.flow["darcy"]:
            if self.flow["advection"]:
                VTKFile("results/" + "liquid_pressure.pvd").write(Function(self.Vp,
                                                                        name="liquid pressure").assign(uviz[self.i_pl]))
            if self.gas_params:
                VTKFile("results/" + "gas_pressure.pvd").write(Function(self.Vp,
                                                                     name="gas pressure").assign(uviz[self.i_pg]))

    def init_solver_parameters(
            self,
            pc_type="lu",
            custom_solver={},
            custom_potential_solver={}):
        if custom_solver:
            PETSc.Sys.Print("*** WARNING: Using custom solver options")
            self.solver_parameters = custom_solver
            self.potential_solver_parameters = custom_potential_solver
            return

        self.potential_solver_parameters = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "l2",
            "snes_rtol": 1e-8,
            "snes_max_it": 20,
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
        }

        num_c = self.num_c
        snes_newtonls = {"snes_type": "newtonls",
                         "snes_linesearch_type": "l2",
                         "snes_monitor": None,
                         "snes_converged_reason": None,
                         "snes_rtol": 1e-16,
                         "snes_atol": 1e-16,
                         "snes_divergence_tolerance": 1e6,
                         "snes_max_it": 50,
                         }

        if pc_type.startswith(("block", "cpr")):

            if pc_type == "block" or pc_type == "cpr":
                mg = {"ksp_type": "preonly",
                      "pc_type": "hypre",
                      }
            elif pc_type == "blockgmg" or pc_type == "cprgmg":
                mg = {"ksp_type": "bcgs",
                      "ksp_rtol": 1e-2,
                      "pc_type": "mg",
                      "mg_coarse_ksp_type": "preonly",
                      "mg_coarse_pc_type": "lu",
                      "mg_levels_ksp_type": "richardson",
                      "mg_levels_ksp_max_it": 1,
                      "mg_levels_pc_type": "bjacobi",
                      "mg_levels_sub_pc_type": "ilu",
                      }
            elif pc_type == "blocklu":
                mg = {"ksp_type": "preonly",
                      "pc_type": "lu",
                      "pc_factor_mat_solver_type": "mumps",
                      }
            else:
                raise NotImplementedError('unrecognized preconditioner type')
            ilu = {"ksp_type": "preonly",
                   "pc_type": "bjacobi",
                   "sub_pc_type": "ilu",
                   }

            if self.flow["porous"]:
                is_list = [str(i + self.num_mass) for i in range(2)]
                U_is = ",".join(is_list)
            else:
                U_is = self.num_mass
            if self.vector_mix:
                U_is = "1,2"
            if self.vector_mix or self.num_mass == 1:
                C_is = 0
            if self.gas_params and self.flow["darcy"] and self.flow["advection"]:
                is_list = [str(self.i_pl), str(self.i_pg)]
                p_is = ",".join(is_list)
                if self.vector_mix:
                    p_is = "3,4"
            is_list = [str(i) for i in range(self.num_mass)]
            C_is = ",".join(is_list)

            if self.num_mass == 1:
                fieldsplit_C = mg
            else:
                fieldsplit_C = {"ksp_type": "preonly",
                                "pc_type": "fieldsplit",
                                "pc_fieldsplit_type": "multiplicative",
                                }
                for i in is_list:
                    fieldsplit_C.update({"fieldsplit_" + i: mg})
        if pc_type.startswith("block"):
            if self.gas_params and self.flow["darcy"] and self.flow["advection"]:
                fieldsplit_p = {"ksp_type": "preonly",
                                "pc_type": "fieldsplit",
                                "pc_fieldsplit_type": "multiplicative",
                                "fieldsplit_0": mg,
                                "fieldsplit_1": mg,
                                }
                fieldsplit_U = {"ksp_type": "preonly",
                                "pc_type": "fieldsplit",
                                "pc_fieldsplit_type": "multiplicative",
                                "fieldsplit_0": mg,
                                "fieldsplit_1": mg,
                                }
                lu = {"ksp_type": "preonly",
                      "pc_type": "lu", }
                pc = {"mat_type": "aij",
                      "ksp_rtol": 1e-10,
                      "ksp_converged_reason": None,
                      "ksp_monitor": None,
                      "ksp_type": "fgmres",
                      "ksp_gmres_restart": 1000,
                      "ksp_max_it": 1000,
                      "ksp_pc_side": "right",
                      "pc_type": "fieldsplit",
                      "pc_fieldsplit_0_fields": U_is,
                      "pc_fieldsplit_1_fields": p_is,
                      "pc_fieldsplit_2_fields": C_is,
                      "pc_fieldsplit_type": "multiplicative",
                      "fieldsplit_0": fieldsplit_U,
                      "fieldsplit_1": fieldsplit_p,
                      "fieldsplit_2": fieldsplit_C,
                      }
            elif self.flow["darcy"]:
                if self.vector_mix:
                    p_is = "3"
                else:
                    p_is = self.i_pl

                fieldsplit_U = {"ksp_type": "preonly",
                                "pc_type": "fieldsplit",
                                "pc_fieldsplit_type": "multiplicative",
                                "fieldsplit_0": mg,
                                "fieldsplit_1": mg,
                                }
                pc_p = mg
                twostage_C = {"ksp_type": "preonly",
                              "pc_type": "composite",
                              "pc_composite_pcs": "bjacobi,fieldsplit",
                              "sub_1": fieldsplit_C,
                              "sub_0_sub_pc_type": "ilu",
                              "sub_0_sub_pc_factor_levels": 0,
                              }

                pc = {"mat_type": "aij",
                      "ksp_rtol": 1e-10,
                      "ksp_converged_reason": None,
                      "ksp_monitor": None,
                      "ksp_type": "fgmres",
                      "ksp_gmres_restart": 1000,
                      "ksp_max_it": 1000,
                      "ksp_pc_side": "right",
                      "pc_type": "fieldsplit",
                      "pc_fieldsplit_0_fields": U_is,
                      "pc_fieldsplit_1_fields": p_is,
                      "pc_fieldsplit_2_fields": C_is,
                      "pc_fieldsplit_type": "multiplicative",
                      "fieldsplit_0": fieldsplit_U,
                      "fieldsplit_1": pc_p,
                      "fieldsplit_2": ilu,  # fieldsplit_C,
                      }
            elif self.flow["porous"]:
                fieldsplit_U = {"ksp_type": "preonly",
                                "pc_type": "fieldsplit",
                                "pc_fieldsplit_type": "additive",
                                "fieldsplit_0": mg,
                                "fieldsplit_1": mg,
                                }

                pc = {"mat_type": "aij",
                      "ksp_rtol": 1e-10,
                      "ksp_converged_reason": None,
                      "ksp_monitor": None,
                      "ksp_type": "fgmres",
                      "ksp_gmres_restart": 200,
                      "ksp_pc_side": "right",
                      "pc_type": "fieldsplit",
                      "pc_fieldsplit_0_fields": C_is,
                      "pc_fieldsplit_1_fields": U_is,
                      "pc_fieldsplit_type": "multiplicative",
                      "fieldsplit_0": ilu,  # fieldsplit_C,
                      "fieldsplit_1": mg,
                      }
            else:
                pc = {"mat_type": "aij",
                      "ksp_rtol": 1e-6,
                      "ksp_converged_reason": None,
                      "ksp_monitor": None,
                      "ksp_type": "fgmres",
                      "ksp_gmres_restart": 200,
                      "ksp_pc_side": "right",
                      "pc_type": "fieldsplit",
                      "pc_fieldsplit_0_fields": C_is,
                      "pc_fieldsplit_1_fields": U_is,
                      "pc_fieldsplit_type": "multiplicative",
                      "fieldsplit_0": fieldsplit_C,
                      "fieldsplit_1": mg,
                      }
        elif pc_type.startswith("cpr"):
            if self.gas_params and self.flow["advection"]:
                fieldsplit_p = {"ksp_type": "preonly",
                                "pc_type": "fieldsplit",
                                "pc_fieldsplit_type": "multiplicative",
                                "fieldsplit_0": mg,
                                "fieldsplit_1": mg,
                                }
                p_fs = "2,3"
            else:
                p_is = str(self.i_pl)
                p_fs = "2"
                fieldsplit_p = mg

            fieldsplit_U = {"ksp_type": "preonly",
                            "pc_type": "fieldsplit",
                            "pc_fieldsplit_type": "multiplicative",
                            "fieldsplit_0": mg,
                            "fieldsplit_1": mg,
                            }
            fieldsplit_U_p = {"ksp_type": "preonly",
                              "pc_type": "fieldsplit",
                              "pc_fieldsplit_type": "multiplicative",
                              "pc_fieldsplit_0_fields": "0,1",  # U_is,
                              "pc_fieldsplit_1_fields": p_fs,  # p_is,
                              "fieldsplit_0": fieldsplit_U,
                              "fieldsplit_1": fieldsplit_p,
                              }
            pc = {"ksp_converged_reason": None,
                  "ksp_monitor": None,
                  "ksp_type": "fgmres",
                  "ksp_rtol": 1e-10,
                  "ksp_gmres_restart": 1000,
                  "ksp_max_it": 1000,
                  "ksp_pc_side": "right",
                  "pc_type": "composite",
                  "pc_composite_type": "multiplicative",
                  "pc_composite_pcs": "fieldsplit,bjacobi",

                  "sub_0_pc_fieldsplit_0_fields": U_is + ',' + p_is,
                  "sub_0_pc_fieldsplit_1_fields": C_is,
                  "sub_0_pc_fieldsplit_type": "additive",
                  "sub_0_fieldsplit_0": fieldsplit_U_p,
                  "sub_0_fieldsplit_1_ksp_type": "gmres",
                  "sub_0_fieldsplit_1_ksp_max_it": 0,
                  "sub_0_fieldsplit_1_pc_type": "none",
                  "sub_1_sub_pc_type": "ilu",
                  "sub_1_sub_pc_factor_levels": 0,
                  "mat_type": "aij",
                  }

        elif pc_type == "lu":
            pc = {"mat_type": "aij",
                  "ksp_type": "preonly",
                  "pc_type": "lu",
                  "pc_factor_mat_solver_type": "mumps",
                  }
        elif pc_type == "ilu":
            pc = {"mat_type": "aij",
                  "ksp_type": "fgmres",
                  "ksp_rtol": 1e-10,
                  "ksp_converged_reason": None,
                  "ksp_monitor": None,
                  "ksp_gmres_restart": 1000,
                  "pc_type": "bjacobi",
                  "sub_pc_type": "ilu",
                  "sub_pc_factor_levels": 0,
                  }
        elif pc_type == "gmg":
            pc = {"mat_type": "aij",
                  "ksp_type": "fgmres",
                  "ksp_gmres_restart": 200,
                  "ksp_converged_reason": None,
                  "ksp_monitor": None,
                  "ksp_rtol": 1e-2,
                  "pc_type": "mg",
                  "mg_coarse_ksp_type": "preonly",
                  "mg_coarse_pc_type": "lu",
                  "mg_levels_ksp_type": "richardson",
                  "mg_levels_ksp_max_it": 1,
                  "mg_levels_pc_type": "bjacobi",
                  "mg_levels_sub_pc_type": "ilu",
                  }
        elif pc_type == "amg":
            pc = {"mat_type": "aij",
                  "ksp_type": "fgmres",
                  "ksp_gmres_restart": 200,
                  "ksp_converged_reason": None,
                  "ksp_monitor": None,
                  "ksp_rtol": 1e-2,
                  "pc_type": "hypre",
                  }

        else:
            raise NotImplementedError('unrecognized preconditioner type')

        solver_parameters = snes_newtonls
        solver_parameters.update(pc)
        self.solver_parameters = solver_parameters

    def setup_bulk_reactions(self):
        """ Setting up the homogeneous bulk reactions using homog_params

        Setup using the law of mass action
        """
        def bulk_reaction(u):
            reactions = [0] * self.num_c
            reaction_f = [0] * len(self.homog_params)
            reaction_b = [0] * len(self.homog_params)
            for idx, reaction in enumerate(self.homog_params):
                reaction_f[idx] = reaction["forward rate constant"]
                if reaction.get("backward rate constant"):
                    reaction_b[idx] = reaction["backward rate constant"]
                elif reaction.get("equilibrium constant"):
                    reaction_b[idx] = reaction["forward rate constant"] / \
                                        reaction["equilibrium constant"]
                if reaction.get("reference concentration"):
                    c_ref = reaction["reference concentration"]
                else:
                    c_ref = 1.0
                for name in reaction["stoichiometry"]:
                    s = reaction["stoichiometry"][name]
                    order = abs(s) # default
                    if reaction.get("reaction order"):
                        if reaction["reaction order"].get(name):
                            order = reaction["reaction order"][name]
                    a = u[self.i_c[name]] / c_ref
                    if s < 0:
                        reaction_f[idx] *= a ** order
                    if s > 0:
                        reaction_b[idx] *= a ** order

            for i in self.idx_c:
                name = self.conc_params[i]["name"]
                for idx, reaction in enumerate(self.homog_params):
                    if name in reaction["stoichiometry"]:
                        s = reaction["stoichiometry"][name]
                        if reaction.get("reference concentration"):
                            c_ref = reaction["reference concentration"]
                        else:
                            c_ref = 1.0
                        reactions[i] += s * c_ref * (reaction_f[idx] - reaction_b[idx])

            if self.flow["porous"]:
                porosity = self.physical_params["porosity"]
                eps_l = porosity * self.Sw # liquid void fraction
                reactions = [eps_l * r for r in reactions]

            return reactions
        if self.physical_params.get("bulk reaction"):
            print = PETSc.Sys.Print
            print("WARNING: Overwriting bulk reaction from physical_params with the one in homog_params")
        self.physical_params["bulk reaction"] = bulk_reaction

    def effective_diffusion(self, D, phase="liquid"):
        """ Bruggeman correlation for effective diffusion in a porous medium
        """
        porosity = self.physical_params["porosity"]
        if phase == "solid":
            vol_frac = (1 - porosity)
        elif phase == "liquid":
            vol_frac = porosity * self.Sw
        elif phase == "gas":
            vol_frac = porosity * (1 - self.Sw)
        return D * vol_frac ** 1.5

    def stefan_maxwell_diffusivity(self, X, u, gas_params):
        sumyD = 0.0
        Mn = self.Mn
        Xj = self.Xj
        name = gas_params["name"]
        for i in range(self.num_g):
            if self.gas_params[i]["name"] is not name:
                j = self.num_liquid + i
                M = self.gas_params[i]["molar mass"]
                D = self.gas_params[i]["diffusion coefficient"][name]
                if self.gas_params[i].get("eliminated") is None:
                    sumyD += u[j] / M / D
                else:
                    sumyD += Xj / M / D
        return (1. - X) / sumyD / Mn

    def knudsen_diffusivity(self, gas_params):
        R = self.physical_params["R"]
        T = self.physical_params["T"]
        rpm = self.physical_params["average pore radius"]
        mass = gas_params["molar mass"]
        return 2. * rpm / 3. * sqrt(8. * R * T / pi / mass)

    def gas_diffusivity(self, X, u, gas_params):
        Dm = self.stefan_maxwell_diffusivity(X, u, gas_params)
        DK = self.knudsen_diffusivity(gas_params)
        return 1 / (1 / Dm + 1 / DK)

    def gas_sources(self, u, v):
        a = 0.0
        source = self.physical_params.get("gas source")
        if source is not None:
            sources = source(u)
            for i in range(self.num_gas):
                if sources[i] != 0.0:
                    a -= sources[i] * v[i + self.num_liquid] * self.dx()
        return a

    def dissolutions(self, u, v, conc_params, gas_params):
        # we assume that the eliminated concentrations don't have dissolution
        a = 0.0
        av = self.physical_params["specific surface area"] * self.Sw
        for gas in gas_params:
            name = gas.get("dissolution")
            if name is not None:
                for c in conc_params:
                    if c["name"] == name:
                        conc = c
                i_c = conc["i_c"]
                i_g = gas["i_g"]
                C_liquid = u[i_c]
                X_gas = u[i_g]
                Hcp = gas["Henry constant"]
                M = gas["molar mass"]
                r_pm = self.physical_params["average pore radius"]
                dTF = r_pm * (1. - (1 - self.Sw)**0.5)
                kGL = conc["diffusion coefficient"] / dTF
                pp_gas = X_gas * self.Mn / M * self.pg
                dissolution = av * kGL * (Hcp * pp_gas - C_liquid)
                a -= dissolution * v[i_c] * self.dx()  # aqueous
                a += M * dissolution * v[i_g] * self.dx()  # gas
                if self.flow["advection"]:
                    # liquid pressure equation
                    a -= M * dissolution * v[self.i_pl] * self.dx()
                # a += M * dissolution * v[self.i_pg] * self.dx() # gas
                # pressure equation
                if self.flow["electroneutrality"]:
                    z = conc["z"]
                    F = self.physical_params["F"]
                    if z != 0:
                        a -= z * F * dissolution * \
                            v[self.num_mass] * self.dx()  # aqueous

        return a

    def relative_permeability(self, S):
        return S**3

    def saturation(self, pl, pg):
        # eye-ball approximation of the saturation curve from Weng Weber
        pc = pl - pg  # capillary pressure
        PC = np.array([-26.75, -20.69, -14.63, -8.56, -2.5,
                      3.57, 9.63, 15.56, 21.97, 27.1]) * 1e3
        S = [0.34, 0.38, 0.44, 0.51, 0.6, 0.69, 0.77, 0.85, 0.91, 0.95]

        Sl = conditional(lt(pc, PC[0]), S[0], 0)
        Sl = conditional(gt(pc, PC[9]), S[9], 0)
        for i in range(9):
            Sl += conditional(And(gt(pc, PC[i]), lt(pc, PC[i + 1])), S[i] + (
                S[i + 1] - S[i]) / (PC[i + 1] - PC[i]) * (pc - PC[i]), 0)
        return Sl

    def set_gas_density(self, X, pg, gas_params):
        R = self.physical_params["R"]
        T = self.physical_params["T"]

        # average molar mass of the mixture
        Mn = 0.
        Xj = 1.0
        for i in range(self.num_g):
            if gas_params[i].get("eliminated") is None:
                Mn += X[i] / gas_params[i]["molar mass"]
                Xj -= X[i]
            else:
                j = i
        Mn += Xj / gas_params[j]["molar mass"]
        Mn = 1. / Mn
        self.Mn = Mn
        self.Xj = Xj

        # ideal gas law
        self.rhog = Mn * pg / R / T * 1e-6  # g/m3 -> g/cm3
        # self.rhog = Constant(0.00182728)

        if self.flow["diffusion"] and (
                self.boundary_markers.get("gas") is not None):
            Mn_gas = 0.0
            for i in range(self.num_g):
                Mn_gas += gas_params[i]["gas"] / gas_params[i]["molar mass"]
            Mn_gas = 1. / Mn_gas
            self.Mn_gas = Mn_gas

    def set_darcy_velocity(self):
        K = self.physical_params["permeability"]
        Sl = self.Sw
        if self.flow["advection"]:
            pl = self.pl
            mu = self.physical_params["liquid viscosity"]
            krl = self.relative_permeability(Sl)
            self.vel = -K / mu * krl * grad(pl)
        if self.gas_params:
            krg = self.relative_permeability(1 - Sl)
            mug = self.physical_params["gas viscosity"]
            pg = self.pg
            self.velg = -K / mug * krg * grad(pg)

    def set_function_spaces(self, layers, family):
        """Set base function spaces

        self.V  : FunctionSpace for concentrations

        self.Vu : FunctionSpace for potentials (if using Poisson or Electroneutrality)

        self.Vp : FunctionSpace for pressures (if using Darcy)
        """

        if family == "DG" and layers is not None:
            # Create function spaces for an extruded mesh
            PETSc.Sys.Print("Creating function spaces for an extruded mesh")
            self.V = FunctionSpace(
                self.mesh,
                "DQ",
                self.poly_degree,
                vfamily="DG",
                vdegree=self.poly_degree)
            self.Vu = FunctionSpace(
                self.mesh,
                "DQ",
                self.poly_degreeU,
                vfamily="DG",
                vdegree=self.poly_degreeU)
            self.Vp = FunctionSpace(
                self.mesh,
                "DQ",
                self.poly_degreep,
                vfamily="DG",
                vdegree=self.poly_degreep)

        else:
            self.V = FunctionSpace(self.mesh, family, self.poly_degree)
            self.Vu = FunctionSpace(self.mesh, family, self.poly_degreeU)
            self.Vp = FunctionSpace(self.mesh, family, self.poly_degreep)

    def mass_conservation_form(
            self,
            C,
            test_fn,
            conc_params,
            u=None):
        """Returns weak form of a mass conservation equation for an aqueous species.

        DG: Using interior penalty for the diffusion term and upwinding for the
        advection-diffusion term.
        """

        n_c = self.num_c
        i_c = conc_params["i_c"]
        D = conc_params.get("diffusion coefficient")
        C_0 = conc_params.get("bulk")
        C_gas = conc_params.get("gas")
        inlet = self.boundary_markers.get("inlet")
        outlet = self.boundary_markers.get("outlet")
        bulk = self.boundary_markers.get("bulk")
        bulk_reaction = self.physical_params.get("bulk reaction")
        bulk_dirichlet = self.boundary_markers.get("bulk dirichlet")
        gas = self.boundary_markers.get("gas")
        if self.flow["porous"]:
            D = self.effective_diffusion(D)

        family = self.family
        mesh = self.mesh
        n = FacetNormal(mesh)
        if family == "DG":
            he = CellVolume(mesh) / FacetArea(mesh)
            D_IP = self.C_IP * max_value(self.penalty_degree**2, 1) / he
            K_IP = -1  # SIP

        a = 0.0
        bcs = []  # Dirichlet BCs for CG
        if bulk_reaction is not None:
            bulk_reaction_term = bulk_reaction(u)[i_c]
            if bulk_reaction_term != 0:
                a -= bulk_reaction_term * test_fn * self.dx()

        if self.flow["poisson"] or self.flow["electroneutrality"] or self.flow["electroneutrality full"]:
            U = u[self.i_Ul]
        if self.flow["migration"]:
            F = self.physical_params["F"]
            z = conc_params["z"]
            R = self.physical_params["R"]
            T = self.physical_params["T"]
            K = D * z * F / R / T
        elif self.echem_params:
            F = self.physical_params["F"]

        if self.flow["diffusion"]:
            # diffusion
            a += inner(D * grad(C), grad(test_fn)) * self.dx()
            if family == "DG":
                # Gamma_I
                a += K_IP * inner(avg(D * grad(C)), jump(test_fn, n)) * self.dS()
                a -= inner(avg(D * grad(test_fn)), jump(C, n)) * self.dS()
                a += avg(D_IP * D) * jump(C) * jump(test_fn) * self.dS()
            # Gamma Inlet
            # bulk_dirichlet = inlet
            if bulk_dirichlet is not None:
                if family == "DG":
                    a += inner(D * (C_0 - C) * n, grad(test_fn)) * \
                        self.ds(bulk_dirichlet)
                    a -= inner(D * grad(C), n * test_fn) * self.ds(bulk_dirichlet)
                    a += inner(D_IP * D * (C - C_0) * n,
                               test_fn * n) * self.ds(bulk_dirichlet)
                # else:
                bcs.append(DirichletBC(self.W.sub(i_c), C_0, bulk_dirichlet))

            if gas is not None and C_gas is not None:
                if family == "DG":
                    a += inner(D * (C_gas - C) * n, grad(test_fn)) * self.ds(gas)
                    a -= inner(D * grad(C), n * test_fn) * self.ds(gas)
                    a += inner(D_IP * D * (C - C_gas) * n,
                               test_fn * n) * self.ds(gas)
                # else:
                bcs.append(DirichletBC(self.W.sub(i_c), C_gas, gas))

            # TODO: define this term outside this function for efficiency?
            if self.flow["finite size"]:
                NA = self.physical_params["Avogadro constant"]
                denominator = 1.0
                numerator = 0.0
                for i in range(n_c):
                    ai = self.conc_params[i].get("solvated diameter")
                    if ai is not None and ai != 0.0:
                        denominator -= NA * ai ** 3 * u[i]
                        numerator += NA * ai ** 3 * grad(u[i])
                a += D * C * inner(numerator / denominator, grad(test_fn)) * self.dx()

        if self.flow["diffusion finite size_BK"]:
            NA = self.physical_params["Avogadro constant"]
            phi = 0.0
            for i in range(n_c):
                ai = self.conc_params[i].get("solvated diameter")
                if ai is not None and ai != 0.0:
                    phi += NA * ai ** 3 * u[i]
            mu_ex = -ln(1 - phi)
            a += D * C * inner(grad(ln(C) + mu_ex), grad(test_fn)) * self.dx()
            if bulk_dirichlet is not None:
                bcs.append(DirichletBC(self.W.sub(i_c), C_0, bulk_dirichlet))

        if self.flow["diffusion finite size_CS"]:
            NA = self.physical_params["Avogadro constant"]
            phi = 0.0
            for i in range(n_c):
                ai = self.conc_params[i].get("solvated diameter")
                if ai is not None and ai != 0.0:
                    phi += NA * ai ** 3 * u[i]
            mu_ex = phi*(8 - 9*phi + 3*phi**2)/(1 - phi)**3
            a += D * C * inner(grad(ln(C) + mu_ex), grad(test_fn)) * self.dx()
            if bulk_dirichlet is not None:
                bcs.append(DirichletBC(self.W.sub(i_c), C_0, bulk_dirichlet))

        if self.flow["diffusion finite size_SP"]:
            NA = self.physical_params["Avogadro constant"]
            phi = 0.0
            for i in range(n_c):
                ai = self.conc_params[i].get("solvated diameter")
                if ai is not None and ai != 0.0:
                    phi += NA * ai ** 3 * u[i]
            mu_ex = -ln(1 - 8*phi)
            a += D * C * inner(grad(ln(C) + mu_ex), grad(test_fn)) * self.dx()
            if bulk_dirichlet is not None:
                bcs.append(DirichletBC(self.W.sub(i_c), C_0, bulk_dirichlet))

        if self.flow["diffusion finite size_SMPNP"]:
            NA = self.physical_params["Avogadro constant"]
            phi = 0.0
            a0 = 2.3e-10    # Size of water molecule
            aj = conc_params.get("solvated diameter")
            betaj = (aj/a0)**3
            for i in range(n_c):
                ai = self.conc_params[i].get("solvated diameter")
                if ai is not None and ai != 0.0:
                    phi += NA * ai ** 3 * u[i]

            # size-modified (SMPNP) correction (cf. Eq. 4 in doi:10.26434/chemrxiv-2022-h2mrp)

            mu_ex = - betaj * ln(1 - phi)

            a += D * C * inner(grad(ln(C) + mu_ex), grad(test_fn)) * self.dx()

            if bulk_dirichlet is not None:
                bcs.append(DirichletBC(self.W.sub(i_c), C_0, bulk_dirichlet))

        if self.flow["diffusion finite size_BMCSL"]:
            NA = self.physical_params["Avogadro constant"]
            psi3 = 0.0
            psi0 = 0.0
            psi1 = 0.0
            psi2 = 0.0
            for i in range(n_c):
                ai = self.conc_params[i].get("solvated diameter")
                if ai is not None and ai != 0.0:
                    psi0 += (pi/6)*(NA * u[i])
                    psi1 += (pi/6)*(NA * ai ** 1 * u[i])
                    psi2 += (pi/6)*(NA * ai ** 2 * u[i])
                    psi3 += (pi/6)*(NA * ai ** 3 * u[i])
            ai = conc_params.get("solvated diameter")

            # Boublik-Mansoori-Carnahan-Sterling-Leland (BMCSL) correction (cf Eq. 4 in doi:10.1016/j.jcis.2007.08.006)

            mu_ex = -(1 + (2*psi2**3*ai**3/(psi3**3)) - (3*psi2**2*ai**2/(psi3**2)))*ln(1 - psi3) \
                + (3*psi2*ai + 3*psi1*ai**2 + psi0*ai**3)/(1 - psi3) \
                + (6*psi1*psi2*ai**3 + 9*psi2**2*ai**2)/(2*(1-psi3)**2) \
                + (3*psi2**3*ai**3)/(1 - psi3)**3 + ((3*psi2**2*ai**2)/(psi3))*((1-3*psi3*0.5)/(1-psi3)**2)  \
                - (psi2**3 * ai**3)*(4*psi3**2 - 5*psi3 + 2)/(psi3**2*(1 - psi3)**3)

            a += D * C * inner(grad(ln(C) + mu_ex), grad(test_fn)) * self.dx()

            if bulk_dirichlet is not None:
                bcs.append(DirichletBC(self.W.sub(i_c), C_0, bulk_dirichlet))

        if self.flow["advection"]:
            vel = self.vel
        # convection and migration
        if self.flow["advection"] or (self.flow["migration"] and z != 0.0):
            if self.flow["advection"] and (
                    self.flow["migration"] and z != 0.0):
                flow = vel - K * grad(U)
            elif self.flow["migration"] and z != 0.0:
                flow = - K * grad(U)
            elif self.flow["advection"]:
                flow = vel
            a -= inner(C * flow, grad(test_fn)) * self.dx()
            if family == "DG":
                # Gamma_I
                flow_N = 0.5 * (dot(flow, n) + abs(dot(flow, n)))
                a += (test_fn('+') - test_fn('-')) * \
                    (flow_N('+') * C('+') - flow_N('-') * C('-')) * self.dS()
            elif self.SUPG:  # SUPG
                # hk = CellDiameter(mesh)
                hk = sqrt(2) * CellVolume(mesh) / CellDiameter(mesh)
                u_norm = inner(flow, flow) ** 0.5
                Pe = hk / 2. * u_norm / D
                # Pe_f = conditional(gt(Pe, 3), 1., Pe/3.) # Wang?
                Pe_f = conditional(gt(Pe, 1), 1. - 1./Pe, 0)  # ElmanSilvesterWathen
                # Pe_f = conditional(gt(Pe, 1), 1., Pe/3) # BrezziMariniRusso
                delta_k = hk / 2. / u_norm * Pe_f
                # p = 1
                a += delta_k * inner(dot(flow, grad(C)), dot(flow, grad(test_fn))) * self.dx()
                if bulk_reaction is not None:
                    if bulk_reaction_term != 0:
                        a -= bulk_reaction_term * delta_k * dot(flow, grad(test_fn)) * self.dx()

            applied = self.boundary_markers.get("applied")
            liquid_applied = self.boundary_markers.get("liquid applied")
            is_applied = applied is not None and not self.flow["porous"]
            is_applied = is_applied or (
                self.boundary_markers.get("liquid applied") is not None)
            idx_app = None
            if is_applied:
                if self.flow["porous"]:
                    idx_app = liquid_applied
                else:
                    idx_app = applied
            if idx_app is not None:
                a += conditional(dot(flow, n) < 0, test_fn
                                 * dot(flow, n) * C_0, 0.0) * self.ds(idx_app)
                a += conditional(dot(flow, n) > 0, test_fn
                                 * dot(flow, n) * C, 0.0) * self.ds(idx_app)
            # Gamma Inlet
            if inlet is not None:
                idx = inlet
                if idx_app is not None:
                    idx = tuple(set(idx) - set(idx_app))
                if C_0 != 0.0:
                    a += conditional(dot(vel, n) < 0, test_fn
                                     * dot(vel, n) * C_0, 0.0) * self.ds(idx)
            # Gamma Outlet
            if outlet is not None:
                idx = outlet
                if idx_app is not None:
                    idx = tuple(set(idx) - set(idx_app))
                a += conditional(dot(vel, n) > 0, test_fn
                                 * dot(vel, n) * C, 0.0) * self.ds(idx)

        # Gamma Bulk
        if bulk is not None and conc_params.get(
                "mass transfer coefficient") is not None:
            a -= conc_params["mass transfer coefficient"] * \
                (C_0 - C) * test_fn * self.ds(bulk)

        # Echem reaction
        name = conc_params["name"]
        if not self.flow["porous"]:
            for echem in self.echem_params:
                electrode = self.boundary_markers.get(echem["boundary"])
                n_ = echem["electrons"]
                if name in echem["stoichiometry"]:
                    nu = echem["stoichiometry"][name]
                    a -= test_fn * nu / n_ / F * \
                        echem["reaction"](u) * self.ds(electrode)
        else:
            av0 = self.physical_params["specific surface area"]
            av = av0 * self.Sw
            for echem in self.echem_params:
                n_ = echem["electrons"]
                if name in echem["stoichiometry"]:
                    nu = echem["stoichiometry"][name]
                    a -= av * test_fn * nu / n_ / F * \
                        echem["reaction"](u) * self.dx()

        # Gamma Neumann (for custom Neumann BC)
        neumann = self.boundary_markers.get("neumann")
        if neumann is not None:
            a -= test_fn * self.neumann(C, conc_params, u) * self.ds(neumann)

        return a, bcs

    def gas_mass_conservation_form(self, X, test_fn, gas_params, u=None):
        """Returns weak form of a mass conservation equation for a gaseous species.

        Using interior penalty for the diffusion term and upwinding for the
        advection tterm.
        """

        i_g = gas_params.get("i_g")
        X_gas = gas_params.get("gas")
        mass = gas_params.get("molar mass")
        vel = self.velg
        rhog = self.rhog
        F = self.physical_params["F"]
        R = self.physical_params["R"]
        T = self.physical_params["T"]
        gas_inlet = self.boundary_markers.get("gas inlet")
        gas_outlet = self.boundary_markers.get("gas outlet")
        gas = self.boundary_markers.get("gas")
        Mn = self.Mn
        if gas is not None:
            Mn_gas = self.Mn_gas

        D = self.gas_diffusivity(X, u, gas_params)
        if self.flow["porous"]:
            D = self.effective_diffusion(D, phase="gas")

        family = self.family
        mesh = self.mesh
        n = FacetNormal(mesh)
        if family == "DG":
            he = CellVolume(mesh) / FacetArea(mesh)
            D_IP = self.C_IP * max_value(self.penalty_degree**2, 1) / he
            K_IP = -1  # SIP

        a = 0.0
        bcs = []

        if self.flow["diffusion"]:
            # diffusion
            a += inner(D * rhog * grad(X), grad(test_fn)) * self.dx()
            if family == "DG":
                # Gamma_I
                a += K_IP * inner(avg(D * rhog * grad(X)),
                                  jump(test_fn, n)) * self.dS()
                a -= inner(avg(D * rhog * grad(test_fn)), jump(X, n)) * self.dS()
                a += avg(D_IP * D * rhog) * jump(X) * jump(test_fn) * self.dS()
            # mixture averaged term
            Davg = rhog * D * X / Mn
            a += inner(Davg * grad(Mn), grad(test_fn)) * self.dx()
            if family == "DG":
                # Gamma_I - what penalization terms do we want here?
                a += K_IP * inner(avg(Davg * grad(Mn)),
                                  jump(test_fn, n)) * self.dS()
                a -= inner(avg(Davg * grad(test_fn)), jump(Mn, n)) * self.dS()
                a += avg(D_IP * Davg) * jump(Mn) * jump(test_fn) * self.dS()
            # Gamma Inlet
            # if inlet is not None:
            #    a += inner(D * rhog * (X_0 - X) * n, grad(test_fn)) * self.ds(inlet)
            #    a -= inner(D * rhog * grad(X), n * test_fn) * self.ds(inlet)
            #    a += inner(D_IP * D * rhog * (X - X_0) * n, test_fn * n) * self.ds(inlet)
            if gas is not None and X_gas is not None:
                if family == "DG":
                    a += inner(D * rhog * (X_gas - X) * n,
                               grad(test_fn)) * self.ds(gas)
                    a -= inner(D * rhog * grad(X), n * test_fn) * self.ds(gas)
                    a += inner(D_IP * D * rhog * (X - X_gas)
                               * n, test_fn * n) * self.ds(gas)

                    a += inner(Davg * (Mn_gas - Mn) * n,
                               grad(test_fn)) * self.ds(gas)
                    a -= inner(Davg * grad(Mn), n * test_fn) * self.ds(gas)
                    a += inner(D_IP * Davg * (Mn - Mn_gas)
                               * n, test_fn * n) * self.ds(gas)
                # else:
                bcs.append(DirichletBC(self.W.sub(i_g), X_gas, gas))
        # convection
        if self.flow["advection"] or self.flow["advection gas only"]:
            flow = rhog * vel
            a -= inner(X * flow, grad(test_fn)) * self.dx()
            if family == "DG":
                # Gamma_I
                flow_N = 0.5 * (dot(flow, n) + abs(dot(flow, n)))
                a += (test_fn('+') - test_fn('-')) * \
                    (flow_N('+') * X('+') - flow_N('-') * X('-')) * self.dS()
            elif self.SUPG:  # SUPG
                hk = CellDiameter(mesh)
                u_norm = inner(flow, flow) ** 0.5
                Pe = hk / 2. * u_norm / D
                delta_k = hk / 2. / u_norm * \
                    conditional(gt(Pe, 1), 1. - 1./Pe, 0)  # Wathen
                # p = 1
                a += delta_k * inner(dot(flow, grad(X)), dot(flow, grad(test_fn))) * self.dx()
            # Gamma Inlet
            if gas_inlet is not None:
                if X_gas != 0.0:
                    a += conditional(dot(flow, n) < 0, test_fn
                                     * dot(flow, n) * X_gas, 0.0) * self.ds(gas_inlet)
            # if gas is not None and X_gas is not None:
            #    a += conditional(dot(flow, n) < 0, test_fn *
            #                     dot(flow, n)*X_gas, 0.0) * self.ds(gas)
            # Gamma Outlet
            if gas_outlet is not None:
                a += conditional(dot(flow, n) > 0, test_fn
                                 * dot(flow, n) * X, 0.0) * self.ds(gas_outlet)
        # Echem reaction
        name = gas_params["name"]
        if not self.flow["porous"]:
            for echem in self.echem_params:
                electrode = self.boundary_markers.get(echem["boundary"])
                n_ = echem["electrons"]
                if name in echem["stoichiometry"]:
                    nu = echem["stoichiometry"][name]
                    a -= test_fn * nu / n_ / F * \
                        echem["reaction"](u) * self.ds(electrode)
        else:
            av0 = self.physical_params["specific surface area"]
            av = av0 * self.Sw
            for echem in self.echem_params:
                n_ = echem["electrons"]
                if name in echem["stoichiometry"]:
                    nu = echem["stoichiometry"][name]
                    a -= mass * av * test_fn * nu / n_ / \
                        F * echem["reaction"](u) * self.dx()

        # Gamma Neumann (for custom Neumann BC)
        neumann = self.boundary_markers.get("neumann")
        if neumann is not None:
            a -= test_fn * self.neumann(X, gas_params, u) * self.ds(neumann)

        return a, bcs

    def charge_conservation_form(self, u, test_fn, conc_params, W=None, i_bc=None):
        """Returns weak form of the charge conservation equation for liquid potential.

        Using interior penalty for potential gradient
        """

        F = self.physical_params["F"]
        R = self.physical_params["R"]
        T = self.physical_params["T"]
        inlet = self.boundary_markers.get("inlet")
        outlet = self.boundary_markers.get("outlet")
        bulk = self.boundary_markers.get("bulk")
        gas = self.boundary_markers.get("gas")
        bulk_reaction = self.physical_params.get("bulk reaction")
        bulk_dirichlet = self.boundary_markers.get("bulk dirichlet")

        family = self.family
        mesh = self.mesh
        n = FacetNormal(mesh)
        if family == "DG":
            he = CellVolume(mesh) / FacetArea(mesh)
            D_IP = self.C_IP * max_value(self.penalty_degree**2, 1) / he
            D_IP2 = self.C_IP * max_value(self.penalty_degreeU**2, 1) / he
            K_IP = -1  # SIP

        a = 0.0
        bcs = []
        K_U = 0.0
        C_n = 0.0  # eliminated concentration

        U = u[self.i_Ul]
        if self.flow["porous"]:
            av0 = self.physical_params["specific surface area"]
            av = av0 * self.Sw
        if bulk_reaction is not None:
            reactions = bulk_reaction(u)

        # Do eliminated concentration last
        for i in self.idx_c + [self.i_el]:
            z = conc_params[i]["z"]
            if z != 0.0:
                C_0 = conc_params[i].get("bulk")
                C_ND = conc_params[i].get("C_ND")
                if C_ND is None:
                    C_ND = 1.0
                C_gas = conc_params[i].get("gas")
                if not i == self.i_el:
                    C = u[conc_params[i]["i_c"]]
                    C_n += z * C * C_ND  # electroneutrality constraint
                else:
                    C = -C_n / z / C_ND
                D = conc_params[i]["diffusion coefficient"]
                if self.flow["porous"]:
                    D = self.effective_diffusion(D)
                K_U += z**2 * F**2 * D * C / R / T * C_ND
                zDF = z * D * F * C_ND
                if self.flow["diffusion"]:
                    # diffusion
                    a += inner(zDF * grad(C), grad(test_fn)) * self.dx()
                    if family == "DG":
                        # Gamma_I
                        a += K_IP * inner(avg(zDF * grad(C)),
                                          jump(test_fn, n)) * self.dS()
                        a -= inner(avg(zDF * grad(test_fn)),
                                   jump(C, n)) * self.dS()
                        # a += avg(D_IP * abs(zDF)) * jump(C) * jump(test_fn) * self.dS()
                        # Gamma Inlet
                        if bulk_dirichlet is not None:
                            a += inner(zDF * (C_0 - C) * n,
                                       grad(test_fn)) * self.ds(bulk_dirichlet)
                            a -= inner(zDF * grad(C), n * test_fn) * \
                                self.ds(bulk_dirichlet)
                            # a += inner(D_IP * abs(zDF) * (C - C_0)
                            #           * n, test_fn * n) * self.ds(inlet)
                        if gas is not None and C_gas is not None:
                            a += inner(zDF * (C_gas - C) * n,
                                       grad(test_fn)) * self.ds(gas)
                            a -= inner(zDF * grad(C), n * test_fn) * self.ds(gas)
                if bulk_reaction is not None:
                    if reactions[i] != 0.0:
                        a -= C_ND * z * F * reactions[i] * test_fn * self.dx()
                # Gamma Bulk
                if bulk is not None and conc_params[i].get(
                        "mass transfer coefficient") is not None:
                    a -= C_ND * z * F * conc_params[i]["mass transfer coefficient"] * \
                        (C_0 - C) * test_fn * self.ds(bulk)
                # echem reaction
                name = conc_params[i]["name"]
                if not self.flow["porous"]:
                    for echem in self.echem_params:
                        electrode = self.boundary_markers.get(
                            echem["boundary"])
                        n_ = echem["electrons"]
                        if name in echem["stoichiometry"]:
                            nu = echem["stoichiometry"][name]
                            a -= C_ND * test_fn * z * nu / n_ * \
                                echem["reaction"](u) * self.ds(electrode)
                else:
                    for echem in self.echem_params:
                        n_ = echem["electrons"]
                        if name in echem["stoichiometry"]:
                            nu = echem["stoichiometry"][name]
                            a -= C_ND * av * test_fn * z * nu / n_ * \
                                echem["reaction"](u) * self.dx()

                # Gamma Neumann (for custom Neumann BC)
                neumann = self.boundary_markers.get("neumann")
                if neumann is not None:
                    a -= C_ND * z * F * test_fn * \
                        self.neumann(C, conc_params[i], u) * self.ds(neumann)

        # diffusion of potential
        a += inner(K_U * grad(U), grad(test_fn)) * self.dx()
        if family == "DG":
            # Gamma_I
            a += K_IP * inner(avg(K_U * grad(U)), jump(test_fn, n)) * self.dS()
            a -= inner(avg(K_U * grad(test_fn)), jump(U, n)) * self.dS()
            a += avg(D_IP2 * K_U) * jump(U) * jump(test_fn) * self.dS()

        is_bulk = (bulk is not None)
        applied = self.boundary_markers.get("applied")
        liquid_applied = self.boundary_markers.get("liquid applied")
        # in porous case, applied is for solid potential
        is_applied = applied is not None and not self.flow["porous"]
        is_liquid_applied = liquid_applied is not None
        dirichlets = []
        if is_bulk:
            dirichlets.append((bulk, Constant(0)))
        if is_applied:
            dirichlets.append((applied, self.U_app))
        if is_liquid_applied:
            dirichlets.append((liquid_applied, self.U_app))
        for diric in dirichlets:
            dirichlet = diric[0]
            U_0 = diric[1]

            if family == "DG":
                a += inner(K_U * (U_0 - U) * n, grad(test_fn)) * self.ds(dirichlet)
                a -= inner(K_U * grad(U), n * test_fn) * self.ds(dirichlet)
                a += inner(D_IP2 * K_U * (U - U_0)
                           * n, test_fn * n) * self.ds(dirichlet)
            # else:
            if i_bc is None:
                bcs.append(DirichletBC(self.W.sub(self.i_Ul), U_0, dirichlet))
            else:
                bcs.append(DirichletBC(W.sub(i_bc), U_0, dirichlet))  # for initial guess

        return a, bcs

    def electroneutrality_form(self, u, test_fn, conc_params):
        """Returns weak form for the electroneutrality condition.

        Only useable for CG.
        """

        n_c = self.num_c
        n_m = self.num_mass
        a = 0.0
        bcs = []

        for i in range(n_c):
            z = conc_params[i].get("z")
            if z != 0:
                a += z * u[i] * test_fn * self.dx()

        bulk = self.boundary_markers.get("bulk")
        is_bulk = (bulk is not None)
        applied = self.boundary_markers.get("applied")
        is_applied = applied is not None and not self.flow["porous"]
        is_applied = is_applied or (
            self.boundary_markers.get("liquid applied") is not None)
        if is_bulk or is_applied:
            if is_applied:
                U_0 = self.U_app
                dirichlet = applied
            elif is_bulk:
                U_0 = Constant(0)
                dirichlet = bulk
            bcs.append(DirichletBC(self.W.sub(n_m), U_0, dirichlet))

        return a, bcs

    def potential_poisson_form(self, u, test_fn, conc_params, solid=False, W=None, i_bc=None):
        """Returns weak form of the Poisson equation for a potential.

        DG: Using interior penalty for potential gradient
        """

        F = self.physical_params["F"]
        R = self.physical_params["R"]
        T = self.physical_params["T"]
        eps_0 = self.physical_params.get("vacuum permittivity")
        eps_r = self.physical_params.get("relative permittivity")
        bulk = self.boundary_markers.get("bulk")
        inlet = self.boundary_markers.get("inlet")
        robin = self.boundary_markers.get("robin")
        poisson_neumann = self.boundary_markers.get("poisson neumann")

        family = self.family
        mesh = self.mesh
        n = FacetNormal(mesh)
        if family == "DG":
            he = CellVolume(mesh) / FacetArea(mesh)
            D_IP = self.C_IP * max_value(self.penalty_degree**2, 1) / he
            D_IP2 = self.C_IP * max_value((self.penalty_degreeU)**2, 1) / he
            K_IP = -1  # SIP

        if self.flow["porous"] and solid:
            i_u = self.i_Us
        else:
            i_u = self.i_Ul
        U = u[i_u]

        if self.flow["porous"]:
            av0 = self.physical_params["specific surface area"]
            av = av0 * self.Sw
            if solid:
                K_U = self.physical_params["solid conductivity"]
                K_U = self.effective_diffusion(K_U, phase="solid")
        if not solid:
            if eps_r is not None and eps_0 is not None:
                K_U = eps_r * eps_0
            elif self.physical_params.get("liquid conductivity"):
                K_U = self.physical_params["liquid conductivity"]
                if self.flow["porous"]:
                    K_U = self.effective_diffusion(K_U, phase="liquid")
            else:
                K_U = 0.0

        a = 0.0
        bcs = []

        C_n = 0.0  # eliminated concentration

        bulk_reaction = self.physical_params.get("bulk reaction")
        if bulk_reaction is not None:
            reactions = bulk_reaction(u)

        if self.flow["electroneutrality"]:
            rang = self.idx_c + [self.i_el]
            i_el = self.i_el
        else:
            i_el = None
            rang = range(self.num_liquid)
        # Do eliminated concentration last
        if not solid:
        # If electroneutral, this one is used for potential initial guess
        # If using Poisson, this includes some things for the "true" Poisson equation
            for i in rang:
                z = conc_params[i].get("z")
                if z is not None:
                    if not i == i_el:
                        C = u[conc_params[i]["i_c"]]
                        C_n += z * C  # electroneutrality constraint
                    else:
                        C = -C_n / z
                    if (bulk_reaction is not None) and (
                            not solid) and self.flow["electroneutrality"]:
                        if reactions[i] != 0.0:
                            a -= z * F * reactions[i] * \
                                test_fn * self.dx(domain=self.mesh)

                    neumann = self.boundary_markers.get("neumann")
                    if (neumann is not None) and (z != 0.0) and self.flow["electroneutrality"]:
                        a -= z * F * test_fn * \
                            self.neumann(C, conc_params[i], u) * self.ds(neumann)
                    if not solid and z != 0:
                        D = conc_params[i]["diffusion coefficient"]
                        if self.flow["porous"]:
                            D = self.effective_diffusion(D)
                        if (eps_r is None or eps_0 is None) and \
                           self.physical_params.get("liquid conductivity") is None:
                            K_U += z**2 * F**2 * D * C / R / T
                        elif z != 0.0 and eps_r:
                            # right-hand side of true Poisson equation
                            a -= F * z * C * test_fn * self.dx()

                        # Echem reaction for charge-conservation eqn
                        if not self.flow["poisson"]:
                            name = conc_params[i]["name"]
                            if not self.flow["porous"]:
                                for echem in self.echem_params:
                                    electrode = self.boundary_markers.get(
                                        echem["boundary"])
                                    n_ = echem["electrons"]
                                    if name in echem["stoichiometry"]:
                                        nu = echem["stoichiometry"][name]
                                        a -= test_fn * z * nu / n_ * \
                                            echem["reaction"](u) * self.ds(electrode)
                            else:
                                for echem in self.echem_params:
                                    n_ = echem["electrons"]
                                    if name in echem["stoichiometry"]:
                                        nu = echem["stoichiometry"][name]
                                        a -= av * test_fn * z * nu / n_ * \
                                            echem["reaction"](u) * self.dx()
        if solid:
            for echem in self.echem_params:
                a += av * test_fn * echem["reaction"](u) * self.dx()
        elif not solid and self.flow["poisson"] and self.flow["porous"]:
            for echem in self.echem_params:
                a -= av * test_fn * echem["reaction"](u) * self.dx()


        # diffusion of potential
        a += inner(K_U * grad(U), grad(test_fn)) * self.dx()
        if family == "DG":
            # Gamma_I
            a += K_IP * inner(avg(K_U * grad(U)), jump(test_fn, n)) * self.dS()
            a -= inner(avg(K_U * grad(test_fn)), jump(U, n)) * self.dS()
            a += avg(D_IP2 * K_U) * jump(U) * jump(test_fn) * self.dS()

        if (bulk_reaction is not None):  # and self.flow["poisson"]:
            if solid:
                n_r = self.num_liquid + 1
            else:
                n_r = self.num_liquid
            if len(reactions) > n_r:
                if reactions[n_r] != 0.0:
                    print("Reactions in solid Poisson. for test only")
                    a -= reactions[n_r] * test_fn * self.dx()

        is_bulk = (bulk is not None) and (not solid)
        applied = self.boundary_markers.get("applied")
        is_applied = applied is not None
        if self.flow["porous"]:
            if solid:
                is_applied = is_applied
            else:
                is_applied = self.boundary_markers.get(
                    "liquid applied") is not None
        dirichlets = []
        if is_bulk:
            dirichlets.append((bulk, Constant(0)))
        if is_applied:
            dirichlets.append((applied, self.U_app))
        for diric in dirichlets:
            dirichlet = diric[0]
            U_0 = diric[1]

            if family == "DG":
                a += inner(K_U * (U_0 - U) * n, grad(test_fn)) * self.ds(dirichlet)
                a -= inner(K_U * grad(U), n * test_fn) * self.ds(dirichlet)
                a += inner(D_IP2 * K_U * (U - U_0)
                           * n, test_fn * n) * self.ds(dirichlet)
            # else:
            if i_bc is None:
                bcs.append(DirichletBC(self.W.sub(i_u), U_0, dirichlet))
            else:
                bcs.append(DirichletBC(W.sub(i_bc), U_0, dirichlet))  # for initial guess

        I_app = self.physical_params.get("applied current density")
        if self.boundary_markers.get("applied solid current") and solid:
            a -= I_app * test_fn * self.ds(self.boundary_markers["applied solid current"])
        if self.boundary_markers.get("applied liquid current") and not solid:
            a -= I_app * test_fn * self.ds(self.boundary_markers["applied liquid current"])

        if robin is not None:
            U_0 = self.U_app
            a -= self.physical_params["gap capacitance"] * \
                (U_0 - U) * test_fn * self.ds(robin)

        if poisson_neumann is not None:
            sigma = self.physical_params["surface charge density"]
            a -= sigma * test_fn * self.ds(poisson_neumann)

        return a, bcs

    def liquid_pressure_form(self, p, test_fn, conc_params, u=None, W=None, i_bc=None):
        """Returns weak form of Pressure equation, i.e. the water mass conservation equation.

        DG: Using interior penalty for pressure gradient
        """

        F = self.physical_params["F"]
        R = self.physical_params["R"]
        T = self.physical_params["T"]
        inlet = self.boundary_markers.get("inlet")
        outlet = self.boundary_markers.get("outlet")
        bulk = self.boundary_markers.get("bulk")
        gas = self.boundary_markers.get("gas")
        gas_inlet = self.boundary_markers.get("gas inlet")

        family = self.family
        mesh = self.mesh
        n = FacetNormal(mesh)
        if family == "DG":
            he = CellVolume(mesh) / FacetArea(mesh)
            D_IP = self.C_IP * max_value(self.penalty_degree**2, 1) / he
            D_IP2 = self.C_IP * max_value((self.penalty_degreep)**2, 1) / he
            D_IP3 = self.C_IP * max_value((self.penalty_degreeU)**2, 1) / he
            K_IP = -1  # SIP

        a = 0.0
        bcs = []
        K_U = 0.0
        C_n = 0.0  # eliminated concentration

        U = u[self.i_Ul]
        av0 = self.physical_params["specific surface area"]
        av = av0 * self.Sw

        K = self.physical_params["permeability"]
        mu = self.physical_params["liquid viscosity"]
        rho = self.physical_params["liquid density"]
        Sl = self.Sw
        krl = self.relative_permeability(Sl)
        K_p = rho * K * krl / mu  # mobility

        # diffusion for pressure
        a += inner(K_p * grad(p), grad(test_fn)) * self.dx()
        if family == "DG":
            # Gamma_I
            a += K_IP * inner(avg(K_p * grad(p)), jump(test_fn, n)) * self.dS()
            a -= inner(avg(K_p * grad(test_fn)), jump(p, n)) * self.dS()
            a += avg(D_IP2 * K_p) * jump(p) * jump(test_fn) * self.dS()

        dirichlet = None
        if gas is not None:
            dirichlet = gas
            if gas_inlet is not None:
                dirichlet += gas_inlet
        else:
            dirichlet = gas_inlet

        if dirichlet is not None:
            p_0 = self.physical_params["p_gas"]
            if family == "DG":
                a += inner(K_p * (p_0 - p) * n, grad(test_fn)) * self.ds(dirichlet)
                a -= inner(K_p * grad(p), n * test_fn) * self.ds(dirichlet)
                a += inner(D_IP2 * K_p * (p - p_0)
                           * n, test_fn * n) * self.ds(dirichlet)
            # else:
            if i_bc is None:
                bcs.append(DirichletBC(self.W.sub(self.i_pl), p_0, dirichlet))
            else:
                bcs.append(DirichletBC(W.sub(i_bc), p_0, dirichlet))  # for initial guess

        # Do eliminated concentration last
        for i in self.idx_c + [self.i_el]:
            z = conc_params[i]["z"]
            mass = conc_params[i].get("molar mass")
            C_0 = conc_params[i].get("bulk")
            C_gas = conc_params[i].get("gas")
            if not i == self.i_el:
                C = u[conc_params[i]["i_c"]]
                C_n += z * C  # electroneutrality constraint
            else:
                C = -C_n / z
            D = conc_params[i]["diffusion coefficient"]
            K_U += mass * z * F * D * C / R / T
            # Gamma Bulk
            if bulk is not None and conc_params[i].get(
                    "mass transfer coefficient") is not None:
                a -= mass * conc_params[i]["mass transfer coefficient"] * \
                    (C_0 - C) * test_fn * self.ds(bulk)
            # Reaction
            if self.flow["porous"]:
                name = conc_params[i]["name"]
                for echem in self.echem_params:
                    n_ = echem["electrons"]
                    if name in echem["stoichiometry"]:
                        nu = echem["stoichiometry"][name]
                        a -= mass * av * test_fn * nu / n_ / \
                            F * echem["reaction"](u) * self.dx()

            # Gamma Neumann (for custom Neumann BC)
            neumann = self.boundary_markers.get("neumann")
            if neumann is not None:
                a -= mass * test_fn * \
                    self.neumann(C, conc_params[i], u) * self.ds(neumann)

        # diffusion of potential
        a += inner(K_U * grad(U), grad(test_fn)) * self.dx()
        if family == "DG":
            # Gamma_I
            a += K_IP * inner(avg(K_U * grad(U)), jump(test_fn, n)) * self.dS()
            a -= inner(avg(K_U * grad(test_fn)), jump(U, n)) * self.dS()
            # a += avg(D_IP3 * K_U) * jump(U) * jump(test_fn) * self.dS()

            is_bulk = (bulk is not None)
            applied = self.boundary_markers.get("applied")
            is_applied = applied is not None and not self.flow["porous"]
            is_applied = is_applied or (
                self.boundary_markers.get("liquid applied") is not None)
            if is_bulk or is_applied:
                if is_applied:
                    U_0 = self.U_app
                    dirichlet = applied
                elif is_bulk:
                    U_0 = Constant(0)
                    dirichlet = bulk
                a += inner(K_U * (U_0 - U) * n, grad(test_fn)) * self.ds(dirichlet)
                a -= inner(K_U * grad(U), n * test_fn) * self.ds(dirichlet)
                # a += inner(D_IP3 * K_U * (U - U_0)
                #           * n, test_fn * n) * self.ds(dirichlet)

        fp = self.physical_params.get("pressure source")
        if fp is not None:
            a -= fp() * test_fn * self.dx()

        return a, bcs

    def gas_mass_conservation_form_pressure(
            self, p, test_fn, gas_params, u=None, W=None, i_bc=None):
        """Returns weak form of the mass conservation equation for a gaseous species.

        DG: Using interior penalty for the pressure diffusion term
        """

        X = self.Xj
        # X_0 = gas_params["bulk"]
        X_gas = gas_params.get("gas")
        mass = gas_params.get("molar mass")
        vel = self.velg
        rhog = self.rhog
        F = self.physical_params["F"]
        R = self.physical_params["R"]
        T = self.physical_params["T"]
        inlet = self.boundary_markers.get("inlet")
        outlet = self.boundary_markers.get("outlet")
        gas = self.boundary_markers.get("gas")
        gas_inlet = self.boundary_markers.get("gas inlet")
        gas_outlet = self.boundary_markers.get("gas outlet")
        Mn = self.Mn
        if gas is not None:
            Mn_gas = self.Mn_gas

        D = self.gas_diffusivity(X, u, gas_params)
        if self.flow["porous"]:
            D = self.effective_diffusion(D, phase="gas")

        family = self.family
        mesh = self.mesh
        n = FacetNormal(mesh)
        if family == "DG":
            he = CellVolume(mesh) / FacetArea(mesh)
            D_IP2 = self.C_IP * max_value(self.penalty_degreep**2, 1) / he
            K_IP = -1  # SIP

        a = 0.0
        bcs = []

        if self.flow["diffusion"]:
            # diffusion
            a += inner(D * rhog * grad(X), grad(test_fn)) * self.dx()
            if family == "DG":
                # Gamma_I
                a += K_IP * inner(avg(D * rhog * grad(X)),
                                  jump(test_fn, n)) * self.dS()
                a -= inner(avg(D * rhog * grad(test_fn)), jump(X, n)) * self.dS()
                # a += avg(D_IP * D * rhog) * jump(X) * jump(test_fn) * self.dS()
            # mixture averaged term
            Davg = rhog * D * X / Mn
            a += inner(Davg * grad(Mn), grad(test_fn)) * self.dx()
            if family == "DG":
                # Gamma_I - what penalization terms do we want here?
                a += K_IP * inner(avg(Davg * grad(Mn)),
                                  jump(test_fn, n)) * self.dS()
                a -= inner(avg(Davg * grad(test_fn)), jump(Mn, n)) * self.dS()
                # a += avg(D_IP * Davg) * jump(Mn) * jump(test_fn) * self.dS()
                if gas is not None and X_gas is not None:
                    a += inner(D * rhog * (X_gas - X) * n,
                               grad(test_fn)) * self.ds(gas)
                    a -= inner(D * rhog * grad(X), n * test_fn) * self.ds(gas)
                    # a += inner(D_IP * D * rhog * (X - X_gas) * n, test_fn * n) * self.ds(gas)

                    a += inner(Davg * (Mn_gas - Mn) * n,
                               grad(test_fn)) * self.ds(gas)
                    a -= inner(Davg * grad(Mn), n * test_fn) * self.ds(gas)
                    # a += inner(D_IP * Davg * (Mn - Mn_gas) * n, test_fn * n) * self.ds(gas)

        # convection
        K = self.physical_params["permeability"]
        mu = self.physical_params["gas viscosity"]
        Sg = 1.0 - self.Sw
        krg = self.relative_permeability(Sg)
        K_p = rhog * X * K * krg / mu  # mobility

        # formulated as elliptic for pressure
        a += inner(K_p * grad(p), grad(test_fn)) * self.dx()
        if family == "DG":
            # Gamma_I
            a += K_IP * inner(avg(K_p * grad(p)), jump(test_fn, n)) * self.dS()
            a -= inner(avg(K_p * grad(test_fn)), jump(p, n)) * self.dS()
            a += avg(D_IP2 * K_p) * jump(p) * jump(test_fn) * self.dS()
        if gas_inlet is not None:
            p_0 = self.physical_params["p_gas"]
            if family == "DG":
                a += inner(K_p * (p_0 - p) * n, grad(test_fn)) * self.ds(gas_inlet)
                a -= inner(K_p * grad(p), n * test_fn) * self.ds(gas_inlet)
                a += inner(D_IP2 * K_p * (p - p_0)
                           * n, test_fn * n) * self.ds(gas_inlet)
            # else:
            if i_bc is None:
                bcs.append(DirichletBC(self.W.sub(self.i_pg), p_0, gas_inlet))
            else:
                bcs.append(DirichletBC(W.sub(i_bc), p_0, gas_inlet))

        # Echem reaction
        name = gas_params["name"]
        if not self.flow["porous"]:
            for echem in self.echem_params:
                electrode = self.boundary_markers.get(echem["boundary"])
                n_ = echem["electrons"]
                if name in echem["stoichiometry"]:
                    nu = echem["stoichiometry"][name]
                    a -= mass * test_fn * nu / n_ / F * \
                        echem["reaction"](u) * self.ds(electrode)
        else:
            av0 = self.physical_params["specific surface area"]
            av = av0 * self.Sw
            for echem in self.echem_params:
                n_ = echem["electrons"]
                if name in echem["stoichiometry"]:
                    nu = echem["stoichiometry"][name]
                    a -= mass * av * test_fn * nu / n_ / \
                        F * echem["reaction"](u) * self.dx()

        # Gamma Neumann (for custom Neumann BC)
        neumann = self.boundary_markers.get("neumann")
        if neumann is not None:
            a -= test_fn * self.neumann(X, gas_params, u) * self.ds(neumann)

        source = self.physical_params.get("gas source")
        if source is not None:
            sources = source(u)
            if sources[self.num_liquid] != 0.0:
                a -= sources[self.num_liquid] * test_fn * self.dx()

        return a, bcs

    def gas_pressure_form(self, p, test_fn, gas_params, u=None):  # currently unused
        """Returns weak form of the gas Pressure equation, i.e. the total gaseous mass
        conservation equation.

        DG: Using interior penalty for the pressure diffusion term.
        """

        F = self.physical_params["F"]
        R = self.physical_params["R"]
        T = self.physical_params["T"]
        inlet = self.boundary_markers.get("inlet")
        outlet = self.boundary_markers.get("outlet")
        bulk = self.boundary_markers.get("bulk")
        gas = self.boundary_markers.get("gas")
        rho = self.rhog

        family = self.family
        mesh = self.mesh
        n = FacetNormal(mesh)
        if family == "degree":
            he = CellVolume(mesh) / FacetArea(mesh)
            D_IP = self.C_IP * max_value(self.penalty_degree**2, 1) / he
            D_IP2 = self.C_IP * max_value((self.penalty_degreep)**2, 1) / he
            K_IP = -1  # SIP

        n_g = self.num_g
        a = 0.0
        bcs = []
        X_n = 0.0  # eliminated concentration

        U = u[self.i_Ul]
        av0 = self.physical_params["specific surface area"]
        av = av0 * self.Sw

        for i in range(n_g):
            mass = gas_params[i].get("molar mass")
            X_gas = gas_params[i].get("gas")
            if i < self.num_gas:
                X = u[i + self.num_liquid]
                X_n += X  # mass fraction constraint
            else:
                X = 1.0 - X_n
            # Reaction
            name = gas_params[i]["name"]
            for echem in self.echem_params:
                n_ = echem["electrons"]
                if name in echem["stoichiometry"]:
                    nu = echem["stoichiometry"][name]
                    a -= mass * av * test_fn * nu / n_ / \
                        F * echem["reaction"](u) * self.dx()

            # Gamma Neumann (for custom Neumann BC)
            neumann = self.boundary_markers.get("neumann")
            if neumann is not None:
                a -= mass * test_fn * \
                    self.neumann(X, gas_params[i], u) * self.ds(neumann)

        K = self.physical_params["permeability"]
        mu = self.physical_params["gas viscosity"]
        Sg = 1.0 - self.Sw
        krg = self.relative_permeability(Sg)
        K_p = rho * K * krg / mu  # mobility

        # diffusion for pressure
        a += inner(K_p * grad(p), grad(test_fn)) * self.dx()
        if family == "DG":
            # Gamma_I
            a += K_IP * inner(avg(K_p * grad(p)), jump(test_fn, n)) * self.dS()
            a -= inner(avg(K_p * grad(test_fn)), jump(p, n)) * self.dS()
            a += avg(D_IP2 * K_p) * jump(p) * jump(test_fn) * self.dS()
        if gas is not None:
            p_0 = self.physical_params["p_gas"]
            if family == "DG":
                a += inner(K_p * (p_0 - p) * n, grad(test_fn)) * self.ds(gas)
                a -= inner(K_p * grad(p), n * test_fn) * self.ds(gas)
                a += inner(D_IP2 * K_p * (p - p_0)
                           * n, test_fn * n) * self.ds(gas)
            # else:
            bcs.append(DirichletBC(self.W.sub(self.i_pg), p_0, gas_inlet))

        fpg = self.physical_params.get("gas pressure source")
        if fpg is not None:
            a -= fpg() * test_fn * self.dx()

        return a, bcs

    def set_boundary_markers(self):
        """Set boundary attributes

        This method should set self.boundary_markers, a :py:class:`dict`, where
        the keys are :py:class:`str:` representing the type of boundary
        condition, and the values are :py:class:`tuple` containing the boundary
        indices. For example :

        .. code-block::

            self.boundary_markers = {"bulk": (1,2,)}

        This would set the boundary condition ``"bulk"`` on boundary indices 1 and 2.

        """
        raise NotImplementedError('method needs to be implemented by solver')

    def set_velocity(self):
        """Set velocity

        This method should set self.vel as a Firedrake vector quantity.
        """
        raise NotImplementedError('method needs to be implemented by solver')

    def neumann(self, C, conc_params, u):
        """ Custom Neumann Boundary condition

        Args:
            C: concentration of species i
            conc_params: conc_params[i], i.e. the concentration parameters of species i
            u: full solution state

        Returns:
            normal flux for species i
        """
        raise NotImplementedError('method needs to be implemented by solver')

    def poisson_neumann(self, u):
        """ Custom Neumann Boundary condition for Poisson
        """
        raise NotImplementedError('method needs to be implemented by solver')


class TransientEchemSolver(EchemSolver):
    """Transient electrochemical model solver.

        Adds accumulation terms and time loop to create a transient model
    """

    def setup_forms(self, us, v):
        """ Setup weak forms for transient case
        """
        print("Using transient form")

        self.dt = Constant(1)
        if not hasattr(self, "time"):
            self.time = Constant(0)
        self.u_old = Function(self.W)
        us_old = split(self.u_old)

        # Backward Euler
        Form, bcs = self.steady_forms(us, v)
        Form_, bcs_ = self.transient_forms(us, v, us_old)
        Form += Form_
        bcs += bcs_

        self.Form = Form
        self.bcs = bcs

    def transient_forms(self, us, v, us_old):
        """ Add transient portion of equations to weak forms
        """
        conc_params = self.conc_params
        gas_params = self.gas_params

        Form = 0
        bcs = []

        print("Adding transient form")
        # mass conservation of aqueous species
        for i in range(self.num_liquid):
            if self.flow["electroneutrality full"] and i == self.num_liquid-1:
                test_fn = v[i+1]  # to avoid zeroes on the diagonal. last species must have a charge
            else:
                test_fn = v[i]
            conc_params[self.idx_c[i]]["i_c"] = i
            a, bc = self.mass_accumulation_form(
                us[i], us_old[i], test_fn, conc_params[self.idx_c[i]], u=us)
            weight = conc_params[self.idx_c[i]].get("residual weight")
            if weight is None:
                weight = 1.0
            Form += weight * a
            bcs += bc

        return Form, bcs

    def mass_accumulation_form(
            self,
            C,
            C_old,
            test_fn,
            conc_params,
            u=None):
        """Returns weak form of a mass accumulation term for an aqueous species.
        """
        print("Adding accum form")

        a = (C - C_old) / self.dt * test_fn * dx()
        return a, []

    def update(self):
        """Overwrite this method to perform a custom operation at each timestep
        """
        pass

    def solve(self, times):
        """Solves the transient problem and outputs the solutions

        Args:
            times (list or :class:`numpy.ndarray`): array of the discrete temporal grid

        """

        self.u_old.assign(self.u)
        if self.save_solutions:
            self.output_state(self.u, time=times[0])
        for i in range(len(times))[1:]:
            self.dt.assign(times[i] - times[i-1])  # calculate timestep
            self.time.assign(times[i])  # this works for Backward Euler
            self.echem_solver.solve()
            self.update()
            self.u_old.assign(self.u)

            if self.save_solutions:
                self.output_state(self.u, initiate=False, time=times[i])
