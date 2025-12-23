from firedrake import *
from echemfem import TransientEchemSolver, EchemSolver, IntervalBoundaryLayerMesh
import numpy as np
import matplotlib.pyplot as plt
from math import log10
import os
import shutil
from scipy.optimize import fsolve

import argparse
import petsc4py
from nonuniform_periodic_grid import nonuniform_periodic_grid
petsc4py.PETSc.Sys.popErrorHandler()
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--family", type=str, default='DG')
args, _ = parser.parse_known_args()

"""
A 1D example of CO2 electrolysis on a Cu cathode using a half cell setup, with time pulsing of the input applied voltage. The diffusion/migration/electroneutrality terms are solved in the bulk electrolyte, and a Tafel description is applied for the surface catalyst reactions.


The setup of the 1D problem follows that of:
Bui, J.C., Kim, C., Weber, A.Z. and Bell, A.T., 2021. Dynamic boundary
layer simulation of pulsed CO2 electrolysis on a copper catalyst. ACS
Energy Letters, 6(4), pp.1181-1188.

Note that nonuniform_periodic_grid() from nonuniform_periodic_grid.py is used to specify the nonuniform time spacing.

"""

#---------------------------------------------------------------------------
# constants and operating conditions
T = 298.15           # temperature (K)
USHE = -1.2
Vcell = Constant(USHE) # applied potential (V vs. SHE)
print(USHE)

# physical constants
R = 8.3144598       # ideal gas constant (J/mol/K)
F = 96485.33289     # Faraday's constant (C/mol)

#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# values used for bulk reactions

# 0.1 M CsHCO3
a_CO2 = 0.034
a_Cs = 0.1

K3 = 4.27e7
K4 = 4.58e3
Kw = 1e-14
K1 = K3 * Kw
K2 = K4 * Kw

# Defining some initial guesses for equilibrium
a_H = 10**(-6.8) # pH = 6.8
# Using K_i = prod(a_product^stoich)/prod(a_reactant^stoich) 
a_OH = Kw / a_H
a_HCO3 = K1 * a_CO2 / a_H
a_CO32 = K2 * a_HCO3 / a_H

C_CO2_bulk = a_CO2 * 1000
C_OH_bulk = a_OH * 1000
C_H_bulk = a_H * 1000
C_Cs_bulk = a_Cs * 1000 
C_HCO3_bulk = C_Cs_bulk + C_H_bulk - C_OH_bulk
C_CO32_bulk = 0.* 1000
#---------------------------------------------------------------------------

# Solving the bicarbonate bulk reaction system + electroneutrality
def reaction_system(u):
# CO2, OH, HCO3, CO3, H, HCOO, HCOOH, Cs
    C_CO2 = C_CO2_bulk
    C_Cs = C_Cs_bulk
    C_OH = u[0]
    C_HCO3 = u[1]
    C_CO3 = u[2]
    C_H = u[3]
    r3 = 1. - C_HCO3/1000. / (K3 * C_OH/1000. * C_CO2/1000.)
    r4 = 1. - C_CO3/1000. / (K4 * C_HCO3/1000. * C_OH/1000.)
    rw = 1. - (C_OH/1000. * C_H/1000.)/Kw
    electro = C_Cs + C_H - C_OH - C_HCO3 - 2 * C_CO3
    return [r3, r4, rw, electro]

Ci = [C_OH_bulk, C_HCO3_bulk, C_CO32_bulk, C_H_bulk]
Copt = fsolve(reaction_system, Ci, xtol=1e-16, maxfev=1000, diag=None)
print(Copt)
print(reaction_system(Copt))
C_OH_bulk = Copt[0]
C_HCO3_bulk = Copt[1]
C_CO32_bulk = Copt[2]
C_H_bulk = Copt[3]

class CarbonateSolver(TransientEchemSolver):
    def __init__(self, period1=1., period2=1.):
        self.time = Constant(0)
        conc_params = []

        #Boundary layer thickness delta (in meters)
        delta = 100e-6
        #Construct mesh, with refined near-catalyst region
        mesh = IntervalBoundaryLayerMesh(400, delta, 200, 5e-7, boundary=(1,2,))
        x = SpatialCoordinate(mesh)[0]              

        homog_params = []

        homog_params.append({"stoichiometry": {"CO2": -1,
                                               "H": 1,
                                               "HCO3": 1},
                             "forward rate constant": 0.04, #1/s
                             "equilibrium constant": K1, #4.27e-7,
                             "reference concentration": 1000.,
                             })
        
        homog_params.append({"stoichiometry": {"HCO3": -1,
                                               "H": 1,
                                               "CO3": 1},
                             "forward rate constant": 56.281, #1/s
                             "equilibrium constant": K2, #4.58e-11,
                             "reference concentration": 1000.,
                             })
        
        homog_params.append({"stoichiometry": {"CO2": -1,
                                               "OH": -1,
                                               "HCO3": 1},
                             "forward rate constant": 2.1e3, #1/s
                             "equilibrium constant": K3, #4.27e7,
                             "reference concentration": 1000.,
                             })
        
        homog_params.append({"stoichiometry": {"HCO3": -1,
                                               "OH": -1,
                                               "CO3": 1},
                             "forward rate constant": 6.5e9, #1/s
                             "equilibrium constant": K4, #4.58e3, # note typo in Bui,
                             "reference concentration": 1000.,
                             })
        
        homog_params.append({"stoichiometry": {"H": 1,
                                               "OH": 1},
                             "forward rate constant": 1.6e-3, #1/s
                             "equilibrium constant": Kw, #1e-14,
                             "reference concentration": 1000.,
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
                            "diffusion coefficient": 1.91E-9,  # m^2/s
                            "bulk": C_CO2_bulk,  # mol/m3
                            "z": 0,
                            })

        conc_params.append({"name": "OH",
                            "diffusion coefficient": 4.93E-9,  # m^2/s
                            "bulk": C_OH_bulk,  # mol/m3
                            "z": -1,
                            })

        conc_params.append({"name": "HCO3",
                            "diffusion coefficient": 1.1E-9,  # m^2/s
                            "bulk": C_HCO3_bulk,  # mol/m3
                            "z": -1,
                            })

        conc_params.append({"name": "CO3",
                            "diffusion coefficient": .801E-9,  # m^2/s
                            "bulk": C_CO32_bulk,  # mol/m3
                            "z": -2,
                            })

        conc_params.append({"name": "H",
                            "diffusion coefficient": 6.95E-9,  # m^2/s
                            "bulk": C_H_bulk,  # mol/m3
                            "z": 1,
                            "residual weight": 1e3,
                            })

        conc_params.append({"name": "HCOO",
                            "diffusion coefficient": 1.46E-9,  # m^2/s
                            "bulk": 0.,  # mol/m3
                            "z": -1,
                            })

        conc_params.append({"name": "HCOOH",
                            "diffusion coefficient": 1.46E-9,  # m^2/s
                            "bulk": 0.,  # mol/m3
                            "z": 0,
                            })

        conc_params.append({"name": "Cs",
                            "diffusion coefficient": 2.17e-9,  # m^2/s
                            "bulk": C_Cs_bulk,  # mol/m3
                            "z": 1,
                            })

        physical_params = {"flow": ["diffusion", "migration", "electroneutrality"],
                   "F": F,  # C/mol
                   "R": R,  # J/K/mol
                   "T": T,  # K
                   "U_app": Vcell,
                   }

        def tafel(U0, i0, alpha, gamma_CO2, gamma_pH):
            def func(u):
                C_CO2 = u[self.i_c["CO2"]]
                C_H = u[self.i_c["H"]]
                Phi_s = self.U_app
                Phi_l = u[self.i_Ul]
                cref = 1000.
                if gamma_CO2 == 0:
                    f_CO2 = 1.
                else:
                    f_CO2 = (C_CO2 / cref) ** gamma_CO2
                if gamma_pH == 0:
                    f_pH = 1.
                else:
                    pH = -ln(C_H / cref) / 2.303
                    f_pH = exp(-gamma_pH * pH)
                eta = Phi_s - Phi_l - U0
                taf = exp(-alpha * F / R / T * eta)

                return i0 * f_CO2 * f_pH * taf
            return func

        echem_params = []
        
        # i0 in A/m2
        # note mistake in i0 values as reported in Bui et al, where cref was set to 1mM instead
        # of 1M to calculate them
        # reaction = tafel(U0, i0, alpha, gamma_CO2, gamma_pH)
        reaction_H2 = tafel(0, 6.36e-1, 0.14, 0, 0.40)
        reaction_CO = tafel(-0.11, 6.3562e2, 0.35, 1.5, 1.56)
        reaction_HCOO = tafel(-0.02, 3.58e1, 0.43, 2., 1.56)
        reaction_CH4 = tafel(0.17, 1.3411e-14, 0.84, 0.84, 1.56)
        reaction_C2H5OH = tafel(0.08, 6.4252e-9, 0.43, 0.96, 0)
        reaction_C2H4 = tafel(0.07, 6.1917e-7, 0.41, 1.36, 0)
        reaction_C3H6O = tafel(0.05, 2.5488e-11, 0.49, 0.96, 0)
        reaction_C3H7OH = tafel(0.09, 7.381e-9, 0.4, 0.96, 0)

        echem_params.append({"reaction": reaction_H2,
                             "electrons": 2,
                             "stoichiometry": {"OH": 2},
                             "boundary": "catalyst",
                             })

        echem_params.append({"reaction": reaction_CO,
                             "electrons": 2,
                             "stoichiometry": {"CO2": -1,
                                               "OH": 2},
                             "boundary": "catalyst",
                             })

        echem_params.append({"reaction": reaction_HCOO,
                             "electrons": 2,
                             "stoichiometry": {"CO2": -1,
                                               "OH": 2,
                                               "HCOO": 1},
                             "boundary": "catalyst",
                             })

        echem_params.append({"reaction": reaction_CH4,
                             "electrons": 8,
                             "stoichiometry": {"CO2": -1,
                                               "OH": 8},
                             "boundary": "catalyst",
                             })

        echem_params.append({"reaction": reaction_C2H5OH,
                             "electrons": 12,
                             "stoichiometry": {"CO2": -2,
                                               "OH": 12},
                             "boundary": "catalyst",
                             })

        echem_params.append({"reaction": reaction_C2H4,
                             "electrons": 12,
                             "stoichiometry": {"CO2": -2,
                                               "OH": 12},
                             "boundary": "catalyst",
                             })

        echem_params.append({"reaction": reaction_C3H6O,
                             "electrons": 16,
                             "stoichiometry": {"CO2": -3,
                                               "OH": 16},
                             "boundary": "catalyst",
                             })

        echem_params.append({"reaction": reaction_C3H7OH,
                             "electrons": 18,
                             "stoichiometry": {"CO2": -3,
                                               "OH": 18},
                             "boundary": "catalyst",
                             })


        super().__init__(conc_params, physical_params, mesh,
                         homog_params=homog_params, echem_params=echem_params,
                         family="CG", p=1)


    def set_boundary_markers(self):
        self.boundary_markers = {"bulk dirichlet": (1,),  #C = C_0
                                 "bulk": (1,), # U_liquid = 0
                                 "catalyst": (2,),
                                 }

    def update(self):
        ## the following lines implement a square wave that oscillates between applied voltages V1 and V2 (vs. SHE); these lines are all evaluated after each timestep
        V1 = Constant(-1.2)
        V2 = Constant(-1.55)
        #find the modulus of time w.r.t. total time period, to allow for repeated behavior over multiple cycles; within each cycle, assign V1 to U_app for time <= period1, and assign V2 to U_app for period1 < time <= period2
        t_mod = self.time.values()[0] % (period1 + period2)
        if t_mod <= period1 + 1e-12:#small tolerance to prevent premature switch caused by floating point imprecision
            self.U_app.assign(V1)
        else:
            self.U_app.assign(V2)
        
        ## print time, applied voltage, and current density (partial and total) values
        print(f"Time = {self.time.values()[0]} s")
        print(f"U_app = {Constant(self.U_app).values()[0]} V")
        
        names = ["H2", "CO", "HCOO", "CH4", "C2H5OH", "C2H4", "C3H6O", "C3H7OH"]
        currents = {}
        for idx, echem in enumerate(solver.echem_params):
            currents[names[idx]] = assemble(echem["reaction"](self.u.subfunctions)*ds(2))

        i_tot = 0.
        for name in currents:
            print("i", name, currents[name] * 0.1, "mA/cm2")
            i_tot += currents[name] * 0.1
        print("i total", i_tot, "mA/cm2")


## specify the pulsing times period1 (corresponding to V1) and period2 (corresponding to V2), in seconds, as argument inputs for CarbonateSolver()
period1 = 1.
period2 = 1.
solver = CarbonateSolver(period1=period1, period2=period2)

## provide a list or array of times to solve, nonuniform time spacing
t_list = nonuniform_periodic_grid(periods=(period1,period2), a0=1e-9, amax=1e-2, refine_window=0.05)

## repeat time spacing for desired number of total cycles
t_list_2 = t_list + (period1+period2)
t_list_3 = t_list + 2.*(period1+period2)

## combine all time intervals:
times = np.unique(np.concatenate([t_list, t_list_2, t_list_3]))

## print time array
print('time array printed:')
print(times)

## Note: if the initial value (V1) is chosen to be sufficiently negative, then a continuation loop may be required, such as: 
# for U in [-1.2, -1.37, -1.55]:
# where the final value is the desired value of V1. In this example, however, -1.2 should work without need of a continuation loop.
for U in [-1.2]:
    solver.U_app.assign(U)
    print("V = %d mV" % np.rint(U * 1000))
    solver.setup_solver(initial_solve=False)
    solver.u_old.assign(solver.u)
    solver.solve(times)