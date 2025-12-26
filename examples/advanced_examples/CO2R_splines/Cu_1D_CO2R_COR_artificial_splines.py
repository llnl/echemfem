from firedrake import *
from echemfem import EchemSolver, IntervalBoundaryLayerMesh
from math import log10
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from petsc4py import PETSc
from cubic_spline_ufl import CubicSplineUFL
from cubic_spline_coeffs import cubic_spline_coeffs, cubic_spline
import argparse
import traceback
import os
print = PETSc.Sys.Print

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--bd_layer", type=float, default=18.9)
parser.add_argument("--CO2_sat", type=float, default=1.)
parser.add_argument("--bicarb_M", type=float, default=0.1)
parser.add_argument("--f_COR", type=float, default=1.)
parser.add_argument("--f_CO2R", type=float, default=1.)

args, _ = parser.parse_known_args()
print(args)

output_dir = f"./output_bd_{args.bd_layer}_CO2_{args.CO2_sat}_bicarb_{args.bicarb_M}_f_COR_{args.f_COR}_f_CO2R_{args.f_CO2R}/"
os.makedirs(output_dir, exist_ok=True)

print(output_dir)

F = 96485.

# equilibrium coefficients
Kw = 1e-8           # mol^2/m^6
K1 = pow(10, -3.37)  # mol/m3
K2 = pow(10, -7.32)  # mol/m3
K3 = K1/Kw
K4 = K2/Kw

# forward rate constants
k1f = 3.71e-2       # s^-1
k2f = 59.44         # s^-1
k3f = 2.23          # m3/(mol*s)
k4f = 6e6           # m3/(mol*s)
kwf = 1.4           # mol/m3/s

# Bulk Electrolyte concentrations
# Approximation of equilibrium from Tom Moore
Keq = K1/K2
C_CO2_max = 34.
C_1_inf = C_CO2_max * args.CO2_sat
C_K = args.bicarb_M * 1000 # 500.
C_3_inf = -C_1_inf*Keq/4 + C_1_inf*Keq/4*sqrt(1+8*C_K/(C_1_inf*Keq))
C_4_inf = (C_K - C_3_inf)/2
C_2_inf = C_3_inf/C_1_inf/K3
C_5_inf = Kw/C_2_inf

C_CO2_bulk = C_1_inf
C_OH_bulk = C_2_inf
C_HCO3_bulk = C_3_inf
C_CO32_bulk = C_4_inf
C_H_bulk = C_5_inf
C_K_bulk = C_K
C_CO_eq = 0.95 # based on solubility
C_CO_bulk = 0. #0.01*C_CO_eq

## Solving the bicarbonate bulk reaction system + electroneutrality
# The above values don't solve this system with very high accuracy, which is
# problematic since this causes a discontinuity at the boundary condition.
# Instead, we fix CO2 and K+ concentrations and solve for the others using law
# of mass action and electroneutrality. Note that two of the reactions are
# linearly dependent on the others (see above how K3 and K4 are obtained from
# the others.
def reaction_system(u):
    C_CO2 = C_CO2_bulk
    C_K = C_K_bulk
    C_OH = u[0]
    C_HCO3 = u[1]
    C_CO3 = u[2]
    C_H = u[3]
    r3 = 1. - C_HCO3 / (K3 * C_OH * C_CO2)
    r4 = 1. - C_CO3 / (K4 * C_HCO3 * C_OH)
    rw = 1. - (C_OH * C_H)/Kw
    electro = C_K + C_H - C_OH - C_HCO3 - 2 * C_CO3
    return [r3, r4, rw, electro]

Ci = [C_OH_bulk, C_HCO3_bulk, C_CO32_bulk, C_H_bulk]
Copt = fsolve(reaction_system, Ci, xtol=1e-16, maxfev=1000, diag=None)
C_OH_bulk = Copt[0]
C_HCO3_bulk = Copt[1]
C_CO32_bulk = Copt[2]
C_H_bulk = Copt[3]

# Cubic spline produced COR data
U_value = -0.8
csv_file_COR = 'COR_fluxes.csv'
coeffs_CH4_COR, aCO_data = cubic_spline_coeffs(csv_file_COR, U_value, 'flux_CH4')
coeffs_C2H4_COR, aCO_data = cubic_spline_coeffs(csv_file_COR, U_value, 'flux_C2H4')
coeffs_H2_COR, aCO_data = cubic_spline_coeffs(csv_file_COR, U_value, 'flux_H2')
flux_CH4_COR = CubicSplineUFL(aCO_data, coeffs_CH4_COR) # TODO: put back CH4
flux_C2H4_COR = CubicSplineUFL(aCO_data, coeffs_C2H4_COR)
flux_H2_COR = CubicSplineUFL(aCO_data, coeffs_H2_COR)
# Cubic spline produced CO2R data
csv_file_CO2R = 'CO2R_fluxes.csv'
coeffs_CO_CO2R, aCO2_data = cubic_spline_coeffs(csv_file_CO2R, U_value, 'flux_CO', a='aCO2')
flux_CO_CO2R = CubicSplineUFL(aCO2_data, coeffs_CO_CO2R)
coeffs_H2_CO2R, aCO2_data = cubic_spline_coeffs(csv_file_CO2R, U_value, 'flux_H2', a='aCO2')
flux_H2_CO2R = CubicSplineUFL(aCO2_data, coeffs_H2_CO2R)
coeffs_HCOO_CO2R, aCO2_data = cubic_spline_coeffs(csv_file_CO2R, U_value, 'flux_HCOO', a='aCO2')
flux_HCOO_CO2R = CubicSplineUFL(aCO2_data, coeffs_HCOO_CO2R)

# changing fluxes by a factor
f_COR = Constant(args.f_COR)
f_CO2R = Constant(args.f_CO2R)

# Define Mesh
delta = args.bd_layer * 1e-6  #50e-6
N_c = 160 #1600 # number of elements
factor = 0.00002 #0.0002 #0.0002 # small element is ratio  of the largest
ratio = factor ** (1 / (N_c - 1))
r = np.array([ratio ** i for i in range(N_c+1)])
r = r-r[0]
r = r/r[-1]
r = r * delta
mesh = IntervalMesh(N_c, delta)
mesh.coordinates.dat.data[:] = r
#mesh = IntervalBoundaryLayerMesh(400, delta, 200, 1e-8) #200 points in delta, 200 in BL
class CarbonateSolver(EchemSolver):
    """
    Firecat with CO2->CO->C2H4
    """
    def __init__(self):

        homog_params = []

        homog_params.append({"stoichiometry": {"CO2": -1,
                                               "H": 1,
                                               "HCO3": 1},
                             "forward rate constant": k1f,
                             "equilibrium constant": K1
                             })

        homog_params.append({"stoichiometry": {"HCO3": -1,
                                               "H": 1,
                                               "CO3": 1},
                             "forward rate constant": k2f,
                             "equilibrium constant": K2
                             })

        homog_params.append({"stoichiometry": {"CO2": -1,
                                               "OH": -1,
                                               "HCO3": 1},
                             "forward rate constant": k3f,
                             "equilibrium constant": K3
                             })

        homog_params.append({"stoichiometry": {"HCO3": -1,
                                               "OH": -1,
                                               "CO3": 1},
                             "forward rate constant": k4f,
                             "equilibrium constant": K4
                             })


        homog_params.append({"stoichiometry": {"OH": 1,
                                               "H": 1},
                             "forward rate constant": kwf,
                             "equilibrium constant": Kw
                             })

        homog_params.append({"stoichiometry": {"HCOOH": -1,
                                               "H": 1,
                                               "HCOO": 1},
                             "forward rate constant": 4.04e5, #1/s
                             "equilibrium constant": 2.05e-4,
                             "reference concentration": 1000.,
                             })

        conc_params = []

        conc_params.append({"name": "CO2",
                            "diffusion coefficient": 2.23e-9,  # m^2/s
                            "bulk": C_CO2_bulk,  # mol/m3
                            "z": 0,
                            "solvated diameter": 0.23e-9
                            })

        conc_params.append({"name": "CO",
                            "diffusion coefficient": 2.23e-9,  # m^2/s
                            "bulk": C_CO_bulk,  # mol/m3
                            "z": 0,
                            "solvated diameter": 0.0
                            })

        conc_params.append({"name": "OH",
                            "diffusion coefficient": 5.29e-9,  # m^2/s
                            "bulk": C_OH_bulk,  # mol/m3
                            "z": -1,
                            "solvated diameter": 0.6e-9
                            })

        conc_params.append({"name": "H",
                            "diffusion coefficient": 9.311e-9,  # m^2/s
                            "bulk": C_H_bulk,  # mol/m3
                            "z": 1,
                            "solvated diameter": 0.56e-9,
                            })

        conc_params.append({"name": "CO3",
                            "diffusion coefficient": 7.0e-10,  # m^2/s
                            "bulk": C_CO32_bulk,  # mol/m3
                            "z": -2,
                            "solvated diameter": 0.788e-9
                            })

        conc_params.append({"name": "HCO3",
                            "diffusion coefficient": 9.4e-10,  # m^2/s
                            "bulk": C_HCO3_bulk,  # mol/m3
                            "z": -1,
			    "solvated diameter": 0.8e-9
                            })

        conc_params.append({"name": "K",
                            "diffusion coefficient": 1.96e-9,  # m^2/s
                            "bulk": C_K_bulk,  # mol/m3
                            "z": 1,
                            "solvated diameter": 8.2e-10 # m
                            })

        conc_params.append({"name": "HCOO",
                            "diffusion coefficient": 1.46E-9,  # m^2/s
                            "bulk": 0.,  # mol/m3
                            "z": -1,
                            "solvated diameter": 3.0e-10 # m
                            })

        conc_params.append({"name": "HCOOH",
                            "diffusion coefficient": 1.46E-9,  # m^2/s
                            "bulk": 0.,  # mol/m3
                            "z": 0,
                            "solvated diameter": 3.0e-10 # m
                            })


        physical_params = {"flow": ["migration", "diffusion", "poisson", "finite size"],
                           "F": 96485.,  # C/mol
                           "R": 8.3144598,  # J/K/mol
                           "T": 273.15 + 25.,  # K
                           "vacuum permittivity": 8.8541878128e-12,  # F/m
                           "relative permittivity": 78.4,
                           "Avogadro constant": 6.02214076e23, #1/mol
                           "U_app":  Constant(-0.5),
                           "gap capacitance": 0.2, # F/m^2
                           "Upzc": 0.16, #V
                           }

        super().__init__(conc_params, physical_params, mesh, homog_params=homog_params, family="CG", p=1)
        snes_newtonls = {"snes_type": "newtonls",
                "snes_linesearch_type": "l2",
                "snes_rtol": 0,
                "snes_atol": 1e-10,
                "snes_stol": 1e-10,
                "snes_divergence_tolerance": "PETSC_UNLIMITED",
                "snes_max_it": 50,
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                }
        self.init_solver_parameters(custom_solver=snes_newtonls)

    def neumann(self, C, conc_params, u):
        name = conc_params["name"]
        aCO = u[self.i_c["CO"]] / C_CO_eq
        aCO2 = u[self.i_c["CO2"]] / C_CO2_max
        if name == "CO2":
            return -flux_CO_CO2R.spline(aCO2) * f_CO2R - flux_HCOO_CO2R.spline(aCO2) * f_CO2R
        elif name == "CO":
            return flux_CO_CO2R.spline(aCO2) * f_CO2R - 2 * flux_C2H4_COR.spline(aCO) * f_COR - flux_CH4_COR.spline(aCO) * f_COR
        elif name == "OH":
            return 2 * flux_H2_CO2R.spline(aCO2) * f_CO2R + 2 * flux_H2_COR.spline(aCO) * f_COR + 2 * flux_CO_CO2R.spline(aCO2) * f_CO2R + 8 * flux_C2H4_COR.spline(aCO) * f_COR + 6 * flux_CH4_COR.spline(aCO) * f_COR + 2 * flux_HCOO_CO2R.spline(aCO2) * f_CO2R
        elif name == "HCOO":
            return flux_HCOO_CO2R.spline(aCO2) * f_CO2R

        else:
            return Constant(0)

    def set_boundary_markers(self):
        self.boundary_markers = {"bulk dirichlet": (1,),  #C = C_0
                                 "bulk": (1,), # U_liquid = 0
                                 "neumann": (2,), 
                                 "robin": (2,)
                                 }


# Initial guess with zero flux
class CarbonateSolverZero(CarbonateSolver):
    def set_boundary_markers(self):
        self.boundary_markers = {"bulk dirichlet": (1,),  #C = C_0
                                 "bulk": (1,), # U_liquid = 0
                                 "robin": (2,)
                                 }
solver0 = CarbonateSolverZero()
solver0.save_solutions = False

print("Initial guess with no echem reactions")
snes_newtonls = {"snes_type": "newtonls",
        "snes_monitor": None,
        "snes_linesearch_type": "l2",
        "snes_converged_reason": None,
        #"ksp_converged_reason": None,
        "snes_rtol": 1e-12,
        "snes_atol": 1e-10,
        "snes_stol": 1e-12, #new addition
        "snes_divergence_tolerance": "PETSC_UNLIMITED", #uncommented
        "snes_max_it": 50,
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        }
solver0.init_solver_parameters(custom_solver=snes_newtonls)
solver0.U_app.assign(U_value - solver0.physical_params["Upzc"])
solver0.setup_solver()
solver0.solve()
solver0.U_app.assign(U_value - solver0.physical_params["Upzc"])
solver0.solve()

# Define Solver
solver = CarbonateSolver()
solver.save_solutions = False
solver.U_app.assign(U_value - solver.physical_params["Upzc"])
solver.setup_solver()#initial_solve=False)
#solver.u.assign(solver0.u)

# array of coordinates, for plotting
x = SpatialCoordinate(solver.mesh)[0]
xs = solver.mesh.coordinates.dat.data # make sure u is CG 1

#initialize arrays for saving data
CO2s = []
COs = []
#COs_total = []
Ks = []
pHs = []
j_H2s_COR = []
j_H2s_CO2R = []
j_H2s = []
j_HCOOs = []
j_COs = []
j_C2H4s = []
j_CH4s = []
j_COs_net = []

#potentials to loop over
dU = -.02
Us = []
U = U_value - dU
min_dU = -0.00001
u_old = Function(solver.W)
while U > -1.52: #-1.4: #-1.0: #-0.54: #-1.04:
    U += dU
    while dU < min_dU:
        try:
            solver.U_app.assign(U - solver.physical_params["Upzc"])
            print("Voltage =", U , "V")
            # updates spline coefficients
            coeffs_CH4_COR, aCO_data = cubic_spline_coeffs(csv_file_COR, U, 'flux_CH4')
            coeffs_C2H4_COR, aCO_data = cubic_spline_coeffs(csv_file_COR, U, 'flux_C2H4')
            coeffs_H2_COR, aCO_data = cubic_spline_coeffs(csv_file_COR, U, 'flux_H2')
            flux_CH4_COR.update_coeffs(coeffs_CH4_COR)
            flux_C2H4_COR.update_coeffs(coeffs_C2H4_COR)
            flux_H2_COR.update_coeffs(coeffs_H2_COR)
            coeffs_CO_CO2R, aCO2_data = cubic_spline_coeffs(csv_file_CO2R, U, 'flux_CO', a='aCO2')
            flux_CO_CO2R.update_coeffs(coeffs_CO_CO2R)
            coeffs_H2_CO2R, aCO2_data = cubic_spline_coeffs(csv_file_CO2R, U, 'flux_H2', a='aCO2')
            flux_H2_CO2R.update_coeffs(coeffs_H2_CO2R)
            coeffs_HCOO_CO2R, aCO2_data = cubic_spline_coeffs(csv_file_CO2R, U, 'flux_HCOO', a='aCO2')
            flux_HCOO_CO2R.update_coeffs(coeffs_HCOO_CO2R)

            u_old.assign(solver.u)
            # solve system
            solver.solve()
            C_CO2, C_CO, C_OH, C_H, C_CO3, C_HCO3, C_K, C_HCOO, C_HCOOH, Phi = solver.u.subfunctions
            success = True
            break
        except Exception as e:
            print(f"Error occurred with dU = {dU}: {e}. Halving dU.")
            traceback.print_exc()
            dU /= 2
            U -= dU
            solver.u.assign(u_old)
    if dU > min_dU:
        break
    solver.output_state(solver.u, prefix=output_dir + '/results/'+str(np.round(U,8))+'/')
    #extract solutions for this potential
    C_CO2, C_CO, C_OH, C_H, C_CO3, C_HCO3, C_K, C_HCOO, C_HCOOH, Phi = solver.u.subfunctions
    pH = -log10(C_H.dat.data[-1] / 1000.)

    aCO = C_CO/C_CO_eq
    aCO2 = C_CO2/C_CO2_max
    j_CO_CO2R_ = 2 * assemble(flux_CO_CO2R.spline(aCO2) * f_CO2R * ds(2)) *F*1e-4*1e3 #mA/cm^2
    j_H2_CO2R_ = 2 * assemble(flux_H2_CO2R.spline(aCO2) * f_CO2R * ds(2)) *F*1e-4*1e3 
    j_HCOO_CO2R_ = 2 * assemble(flux_HCOO_CO2R.spline(aCO2) * f_CO2R * ds(2)) *F*1e-4*1e3 
    j_C2H4_COR_ = 8 * assemble(flux_C2H4_COR.spline(aCO) * f_COR * ds(2)) *F*1e-4*1e3 
    j_CH4_COR_ = 6 * assemble(flux_CH4_COR.spline(aCO) * f_COR * ds(2)) *F*1e-4*1e3 
    j_H2_COR_ = 2 * assemble(flux_H2_COR.spline(aCO) * f_COR * ds(2)) *F*1e-4*1e3 
    j_CO_net = j_CO_CO2R_ - j_C2H4_COR_/8 * 2 * 2 - j_CH4_COR_/6 * 2
    j_H2_net = j_H2_COR_ + j_H2_CO2R_

    aCO = C_CO.dat.data[-1]/C_CO_eq
    aCO2 = C_CO2.dat.data[-1]/C_CO2_max
    print("aCO =", aCO)
    print("aCO2 =", aCO2)
    #store quantities of interest into arrays
    Us.append(round(U,5))
    CO2s.append(C_CO2.dat.data[-1])
    COs.append(C_CO.dat.data[-1])
    Ks.append(C_K.dat.data[-1])
    pHs.append(pH)
    j_H2s_COR.append(j_H2_COR_)
    j_H2s_CO2R.append(j_H2_CO2R_)
    j_H2s.append(j_H2_net)
    j_HCOOs.append(j_HCOO_CO2R_)
    j_COs.append(j_CO_CO2R_)
    j_COs_net.append(j_CO_net)
    j_C2H4s.append(j_C2H4_COR_)
    j_CH4s.append(j_CH4_COR_)
    print("j_H2_COR_     =", j_H2_COR_,     "mA/cm2")
    print("j_H2_CO2R_    =", j_H2_CO2R_,    "mA/cm2")
    print("j_CO_CO2R_    =", j_CO_CO2R_,    "mA/cm2")
    print("j_C2H4_COR_   =", j_C2H4_COR_,   "mA/cm2")
    print("j_CH4_COR_    =", j_CH4_COR_,    "mA/cm2")
    print("j_HCOO_CO2R_  =", j_HCOO_CO2R_,  "mA/cm2")

    print(
        "total current =",
        j_H2_COR_ + j_H2_CO2R_ + j_CO_CO2R_ + j_C2H4_COR_ + j_CH4_COR_ + j_HCOO_CO2R_,
        "mA/cm2"
    )

    # plot
    us = solver.u.subfunctions
    for i, u in enumerate(us):
        if i == solver.i_Ul: # Phi_l
            name = "Phi"
            units = "V"
        else:
            name = solver.conc_params[i]["name"]
            units = "mol/m3"
        U_str = str(np.round(U,8))
        plt.figure()
        plt.plot(xs*1e6, u.dat.data, label=f'{name} vs x')
        if name == "H" or name == "OH":
            plt.yscale('log')
        plt.xlabel("x (um)")
        plt.ylabel(f"{name} ({units})")
        plt.title(f"Plot of {name} vs x at {U_str}")
        plt.legend()
        plt.grid(True)

        plt.savefig(output_dir + f"plot_{U_str}_{name}.png")  
        plt.close() 

Us = np.array(Us)

#save data
np.savetxt(output_dir + 'CO2_interface.tsv',CO2s)
np.savetxt(output_dir + 'CO_interface.tsv',COs)
np.savetxt(output_dir + 'CO_total_interface.tsv',COs)
np.savetxt(output_dir + 'K_interface.tsv',Ks)
np.savetxt(output_dir + 'j_H2s_CO2R.tsv',j_H2s_CO2R)
np.savetxt(output_dir + 'j_H2s_COR.tsv',j_H2s_COR)
np.savetxt(output_dir + 'j_H2s.tsv',j_H2s)
np.savetxt(output_dir + 'j_COs.tsv',j_COs)
np.savetxt(output_dir + 'j_COs_net.tsv',j_COs_net)
np.savetxt(output_dir + 'j_C2H4s.tsv',j_C2H4s)
np.savetxt(output_dir + 'j_CH4s.tsv',j_CH4s)
np.savetxt(output_dir + 'U_SHE.tsv', Us)
np.savetxt(output_dir + 'pH_interface.tsv',pHs)

## Plotting

# Voltage values
#selected_voltages = [-0.5, -0.52]#, -0.56, -0.6, -0.66, -0.7, -0.76, -0.8, -0.86, -0.9, -0.96, -1.0, -1.04]
#selected_voltages = [-0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5]#, -1.04]
#selected_voltages = [-0.5, -0.56, -0.6, -0.66, -0.7, -0.76, -0.8, -0.86, -0.9, -0.96, -1.0, -1.06, -1.1, -1.16, 1.2, 1.26]#, -1.04]
selected_voltages = Us
selected_voltages = [x for x in selected_voltages if x >= Us[-1]]
print(selected_voltages)
selected_indices = [np.where(Us == v)[0][0] for v in selected_voltages]
print(selected_indices)
# Extract current densities at selected voltages
j_H2s_COR_selected = [j_H2s_COR[i] for i in selected_indices]
j_H2s_CO2R_selected = [j_H2s_CO2R[i] for i in selected_indices]
j_COs_selected = [j_COs[i] for i in selected_indices]
j_C2H4s_selected = [j_C2H4s[i] for i in selected_indices]
j_CH4s_selected = [j_CH4s[i] for i in selected_indices]
# FE stacked bar chart for all reactions

# Combine data for stacked bar chart
reactions = ["H2 COR", "H2 CO2R", "CO", "C2H4", "CH4"]

all_current_densities = np.array([
    j_H2s_COR_selected,
    j_H2s_CO2R_selected,
    j_COs_selected,
    j_C2H4s_selected,
    j_CH4s_selected
])
print(all_current_densities)

# Calculate faradaic efficiency (normalize to total current)
total_current = np.sum(all_current_densities, axis=0)
faradaic_efficiencies = all_current_densities / total_current
print(faradaic_efficiencies)

# Plot stacked bar chart
x = np.arange(len(selected_voltages))  # Indices for the bar positions
width = 0.8  # Width of bars

fig, ax = plt.subplots(figsize=(10, 6))
bottoms = np.zeros(len(selected_voltages))
for i, (reaction, efficiencies) in enumerate(zip(reactions, faradaic_efficiencies)):
    ax.bar(x, efficiencies, width, label=reaction, bottom=bottoms)
    bottoms += efficiencies

# Customize plot
ax.set_xticks(x)
ax.set_xticklabels([f"{v:.2f} V" for v in selected_voltages])
ax.set_xlabel("Voltage (V)")
ax.set_ylabel("Faradaic Efficiency")
ax.set_title("Faradaic Efficiency for Each Reaction at Selected Voltages")
ax.legend(title="Reactions")
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Save and show the plot
plt.tight_layout()
plt.savefig(output_dir + "faradaic_efficiency_stacked.png")

# Total Current Plot

plt.figure()
plt.plot(x, total_current, marker='x', linestyle='-', color='red')
plt.xlabel("Voltage (V)")
plt.ylabel("Total Current Density (mA/cm2)")
plt.title("Total Current Density vs Voltage")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir + "total_current_density.png")

# Net FE stacked bar chart

# Combine data for stacked bar chart
reactions = ["H2", "CO", "C2H4", "CH4"]

all_current_densities = np.array([
    [x + y for x, y in zip(j_H2s_COR_selected,j_H2s_CO2R_selected)],
    [x - 2*2/8 * y - 2/6*z for x,y,z in zip(j_COs_selected, j_C2H4s_selected, j_CH4s_selected)],
    j_C2H4s_selected,
    j_CH4s_selected
])

# Calculate faradaic efficiency (normalize to total current)
total_current = np.sum(all_current_densities, axis=0)
faradaic_efficiencies = all_current_densities / total_current

# Plot stacked bar chart
x = np.arange(len(selected_voltages))  # Indices for the bar positions
width = 0.8  # Width of bars

fig, ax = plt.subplots(figsize=(10, 6))
bottoms = np.zeros(len(selected_voltages))
for i, (reaction, efficiencies) in enumerate(zip(reactions, faradaic_efficiencies)):
    ax.bar(x, efficiencies, width, label=reaction, bottom=bottoms)
    bottoms += efficiencies

# Customize plot
ax.set_xticks(x)
ax.set_xticklabels([f"{v:.2f} V" for v in selected_voltages])
ax.set_xlabel("Voltage (V)")
ax.set_ylabel("Faradaic Efficiency")
ax.set_title("Net Faradaic Efficiency for Total Production at Selected Voltages")
ax.legend(title="Products")
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Save and show the plot
print("Plotting")
plt.tight_layout()
plt.savefig(output_dir + "net_faradaic_efficiency_stacked.png")

# CO2R FE

# Combine data for stacked bar chart
reactions = ["H2", "CO"]

all_current_densities = np.array([
    j_H2s_CO2R_selected,
    j_COs_selected,
])

# Calculate faradaic efficiency (normalize to total current)
total_current = np.sum(all_current_densities, axis=0)
faradaic_efficiencies = all_current_densities / total_current

# Plot stacked bar chart
x = np.arange(len(selected_voltages))  # Indices for the bar positions
width = 0.8  # Width of bars

fig, ax = plt.subplots(figsize=(10, 6))
bottoms = np.zeros(len(selected_voltages))
for i, (reaction, efficiencies) in enumerate(zip(reactions, faradaic_efficiencies)):
    ax.bar(x, efficiencies, width, label=reaction, bottom=bottoms)
    bottoms += efficiencies

# Customize plot
ax.set_xticks(x)
ax.set_xticklabels([f"{v:.2f} V" for v in selected_voltages])
ax.set_xlabel("Voltage (V)")
ax.set_ylabel("Faradaic Efficiency")
ax.set_title("CO2R Faradaic Efficiency at Selected Voltages")
ax.legend(title="Products")
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Save and show the plot
plt.tight_layout()
plt.savefig(output_dir + "faradaic_efficiency_stacked_CO2R.png")

plt.figure()
plt.plot(Us, CO2s, marker='o', linestyle='-', label="CO2")
plt.xlabel("Voltage (V)")
plt.ylabel("CO2 (mM)")
plt.title("CO2 vs Voltage")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir + "CO2_vs_V.png")

plt.figure()
plt.plot(Us, COs, marker='s', linestyle='-', label="CO", color='red')
plt.xlabel("Voltage (V)")
plt.ylabel("CO")
plt.title("CO vs Voltage")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir + "CO_vs_V.png")
