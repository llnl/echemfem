from firedrake import *
from echemfem import EchemSolver
import numpy as np
"""
A symmetric cylindrical pore model for CO2 electrolysis using electroneutral
Nernst-Planck and simplified bicarbonate bulk reactions.

Elucidating Mass Transport Regimes in Gas Diffusion Electrodes for CO2 Electroreduction
Thomas Moore, Xiaoxing Xia, Sarah E. Baker, Eric B. Duoss, and Victor A. Beck
ACS Energy Letters 2021 6 (10), 3600-3606
"""
# operating conditions
T = 293.15             # temperature (K)
Vcell = Constant(-0.0)  # potential (V)

# physical constants
R = 8.3144598       # ideal gas constant (J/mol/K)
F = 96485.33289     # Faraday's constant (C/mol)

# CO2RR
i0CO2RR = 4.71e-4 * 10   # exchange current density (A/m2)
alphacCO2RR = 0.44    # cathode coefficient
U0CO2RR = -0.11   # standard potential (V)

# HER
i0HER = 1.16e-6 * 10  # exchange current density (A/m2)
alphacHER = 0.36     # cathode coefficient
U0HER = 0.0         # standard potential (V)

# bulk reaction
k2 = 2.19e3 * 1e-3  # m^3/mol/s

# electrolyte properties
cref = 1e3          # reference concentration (mol/m3)

# diffusion coefficients (m2/s)
# [CO2 OH- CO32 K+ CO H2]
D = [1.910e-9, 5.29e-9, 0.92e-9, 1.96e-9, 2.03e-9, 4.5e-9]

# charges
z = [0, -1, -2, 1, 0, 0]

# Bulk Electrolyte concentrations
cKb = 1000                # mol/m3
c0OHb = 1000
c0CO3b = 0.
c0CO2b = 0.
c0COb = 0.
c0H2b = 0.
cb = np.array([c0CO2b, c0OHb, c0CO3b, cKb, c0COb, c0H2b])
H_CO2 = 0.015e3
H_H2 = 0.
H_CO = 0.


class PorousSolver(EchemSolver):
    def __init__(self):

        def bulk_reaction(y):
            i_CO2 = self.i_c["CO2"]
            i_OH = self.i_c["OH"]
            i_CO3 = self.i_c["CO3"]
            yCO2 = y[i_CO2]
            yOH = y[i_OH]
            rCO2 = - k2 * yCO2 * yOH
            rOH = 2 * rCO2
            rCO3 = - rCO2
            rxns = [0.] * self.num_c
            rxns[i_CO2] = rCO2
            rxns[i_OH] = rOH
            rxns[i_CO3] = rCO3
            return rxns

        mesh = RectangleMesh(100, 500, 2e-6, 10e-6, quadrilateral=True)
        _, Z = SpatialCoordinate(mesh)
        active = conditional(le(Z, 5e-6), 1., 0.)
        conc_params = []

        conc_params.append({"name": "CO2",
                            "diffusion coefficient": D[0],
                            "z": z[0],
                            "bulk": cb[0],
                            "gas": H_CO2,
                            })

        conc_params.append({"name": "OH",
                            "diffusion coefficient": D[1],
                            "z": z[1],
                            "bulk": cb[1],
                            })

        conc_params.append({"name": "CO3",
                            "diffusion coefficient": D[2],
                            "z": z[2],
                            "bulk": cb[2],
                            })

        conc_params.append({"name": "K",
                            "diffusion coefficient": D[3],
                            "z": z[3],
                            "bulk": cb[3],
                            "eliminated": True,
                            })

        conc_params.append({"name": "CO",
                            "diffusion coefficient": D[4],
                            "z": z[4],
                            "bulk": cb[4],
                            "gas": H_CO,
                            })

        conc_params.append({"name": "H2",
                            "diffusion coefficient": D[5],
                            "z": z[5],
                            "bulk": cb[5],
                            "gas": H_H2,
                            })

        physical_params = {"flow": ["diffusion", "migration", "electroneutrality"],
                           "F": F,  # C/mol
                           "R": R,  # J/K/mol
                           "T": T,  # K
                           "U_app": Vcell,  # V
                           "bulk reaction": bulk_reaction,
                           }

        def reaction_CO2RR(u):
            CCO2 = u[self.i_c["CO2"]]
            Phi2 = u[self.i_Ul]
            Phi1 = physical_params["U_app"]
            UCO2RR = U0CO2RR
            etaCO2RR = Phi1 - Phi2 - UCO2RR  # reaction overpotential (V)
            iCO2RR = i0CO2RR * CCO2 / cref * exp(-((alphacCO2RR * F) / (R * T)) * etaCO2RR)
            return active * iCO2RR

        def reaction_HER(u):
            Phi2 = u[self.i_Ul]
            Phi1 = physical_params["U_app"]
            UHER = U0HER
            etaHER = Phi1 - Phi2 - UHER  # reaction overpotential (V)
            iHER = i0HER * exp(-((alphacHER * F) / (R * T)) * etaHER)
            return active * iHER
        echem_params = []

        echem_params.append({"reaction": reaction_CO2RR,
                             "electrons": 2,
                             "stoichiometry": {"CO2": -1,  # reactant
                                               "OH": 2,
                                               "CO": 1},  # product
                             "boundary": "catalyst",
                             })

        echem_params.append({"reaction": reaction_HER,
                             "electrons": 2,
                             "stoichiometry": {"OH": 2,
                                               "H2": 1},  # product
                             "boundary": "catalyst",
                             })

        super().__init__(conc_params, physical_params, mesh, echem_params=echem_params, family="CG", coordinates="cylindrical")

    def set_boundary_markers(self):
        self.boundary_markers = {"gas": (3,),  # C = C_gas
                                 "bulk": (4,),  # V = 0
                                 "bulk dirichlet": (4,),  # C = C_bulk
                                 "catalyst": (2,),  # CO2R
                                 }


solver = PorousSolver()
solver.setup_solver(initial_solve=False)
solver.save_solutions = False
solver.solve()

## Plotting

# getting active region
_, Z = SpatialCoordinate(solver.mesh)
active = conditional(le(Z, 5e-6), 1., 0.)

def get_boundary_dofs(V, i):
    u = Function(V)
    bc = DirichletBC(V, active, i)
    bc.apply(u)
    return np.where(u.vector()[:] == 1)

dofs = get_boundary_dofs(solver.V, 2)
Z_cat = Function(solver.V).interpolate(Z).dat.data[dofs]

import matplotlib.pyplot as plt
Vlist = np.linspace(-0.71, -1.31, num=13)
for Vs in Vlist:
    solver.U_app.assign(Vs)
    print("V = %d mV" % np.rint(Vs * 1000))
    solver.solve()
    # [CO2 OH- CO32- CO H2]
    cCO2, cOH, cCO3, cCO, cH2, phi2 = solver.u.subfunctions
    cK = Function(solver.V).assign(2 * cCO3 + cOH)

    filename = "cylindrical_pore/CO2_%dmV.png" % np.rint(-Vs * 1000)
    fig, axes = plt.subplots()
    levels = np.linspace(0, H_CO2, 41)
    contours = tricontourf(cCO2, levels=levels, axes=axes, cmap="turbo")
    fig.colorbar(contours)
    plt.axis('scaled')
    plt.savefig(filename)
    plt.clf()

    filename = "cylindrical_pore/catalyst_CO2_%dmV.png" % np.rint(-Vs * 1000)
    fig, axes = plt.subplots()
    plt.plot(Z_cat, cCO2.dat.data[dofs])
    plt.savefig(filename)
    plt.clf()
