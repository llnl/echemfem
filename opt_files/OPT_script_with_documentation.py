####-------------for OPTIMIZATION and other functionality: import needed packages
from firedrake import *
from echemfem import EchemSolver, RectangleBoundaryLayerMesh
import numpy as np
import matplotlib.pyplot as plt
from math import log10
import os
import shutil

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from firedrake_adjoint import *
from pyMMAopt import MMASolver
from pyMMAopt import ReducedInequality

from scipy.optimize import fsolve

import sys
##--------------------

####--------added to eliminate extraneous parallel printing of certain statements to output file
from petsc4py import PETSc
print = PETSc.Sys.Print
##--------------------

n_sections = int(sys.argv[1]) #number of sections
print('num sections: ',n_sections)
objective = str(sys.argv[2]) #choice of objective, either "current" to maximize C2H4 current density or "FE" to maximize C2H4 Faradaic efficiency
print('Objective: ', objective)
V_final = float(sys.argv[3]) #final applied potential (V vs. SHE); positive value is given, but is understood to be negated (i.e., 1.35 implies V_final = -1.35 V vs. SHE)
print('V final: ', V_final)
flow_rate = float(sys.argv[4]) #flow rate; for now, only 3.0ml/min and 30.0ml/min are implemented, but this can be extended to intermediate and other flow rates if needed
print('flow rate (ml/min): ',flow_rate)
# Check if flow_rate is either 3.0 or 30.0
if flow_rate not in [3.0, 30.0]:
    raise ValueError(f"Invalid flow rate: {flow_rate}. Flow rate must be either 3.0 or 30.0.")

####---------- operating conditions
T = 298. # temperature (K)
Vcell = Constant(-1.35) # initial applied potential (V vs. SHE)
##----------------

####---------- physical constants
R = 8.3144598       # ideal gas constant (J/mol/K)
F = 96485.33289     # Faraday's constant (C/mol)
##----------------

####----------- other reference values
cref = 1.
cref_Ag = 1.e3
##---------------

####-------------------these parameters are for Ag Tafel expressions
U0_CO = -0.11    # standard potential (V vs. SHE)
U0_H2_Ag = 0.0         # standard potential (V vs. SHE)


####-----------Sechenov coefficients for CO2 activity
h_s_OH = 8.39E-5 #m^3/mol
h_s_HCO3 = 9.67E-5 #m^3/mol ##!!!note that this is reported as 5.49E-5 in https://jpldataeval.jpl.nasa.gov/pdf/Jpl15_Sectn5_HeterogenChem.pdf
h_s_CO3 = 14.23E-5 #m^3/mol
h_s_H = 0. #m^3/mol
h_s_K = 9.2E-5 #m^3/mol ##reported as 9.22E-5 (different # of sig figs) in Zeng et al. ACS Catalysis 2020 https://doi.org/10.1021/acscatal.9b05272
h_g_CO2 = -1.7159E-5 #m^3/mol
##-------------

####-------------constants for Ag Tafel expressions
#for CO
i0_CO = 1.905E-6 #5.495E-4 ##A/m^2
alpha_c_CO = 0.544 
gamma_CO2_CO = -(2-alpha_c_CO)/2
gamma_OH_CO = alpha_c_CO

#for H2
i0_H2_Ag = 1.698E-6 ##A/m^2
alpha_c_H2_Ag = 0.312 
gamma_OH_H2 = alpha_c_H2_Ag
##-------------

##---------end of Ag Tafel expression parameters


####-------------------------------------------these parameters here are for Cu Tafel expressions, with fitted Tafel slopes (using data from Li et al. Nat. Comm. 2021), rather than with the Tafel slope values directly reported in the reference --------------------------------------------------------------------------
#only pH = 7.2 data is used in this study, and so only this set of parameters is reported here. Data for other pH's can be obtained from Li et al. Nat. Comm. 2021
#pH of data used in fit
pH = 7.2

# C2H4
i0_C2H4 = 1.3835E-11 #(A/m^2)
alpha_c_C2H4 = 0.4990
UC2H4 = 0.17 #- 2.303*R*T*pH/F #(V vs. SHE)

# C2H6O
i0_C2H6O = 8.9302E-11 #(A/m^2)
alpha_c_C2H6O = 0.4452
UC2H6O = 0.19 #- 2.303*R*T*pH/F #(V vs. SHE)

# H2
i0_H2_Cu = 4.6270E-5 #(A/m^2)
alpha_c_H2_Cu = 0.2547
UH2 = 0. #- 2.303*R*T*pH/F #(V vs. SHE)

# CH4
i0_CH4 = 2.0064E-11#(A/m^2)
alpha_c_CH4 = 0.4731
UCH4 = 0.26 #- 2.303*R*T*pH/F #(V vs. SHE)
##---------end of Cu Tafel expression parameters---------------------------



####-------------mesh/domain parameters start-------------------
#2 refinement levels in y are used (inner and outer)

La = Constant(0.0011) #inlet section x-length (meters)
Lx = Constant(0.011) #catalyst section x-length (meters)
if flow_rate == 3.0:
    Ly = Constant(0.0006) #domain height (y-length) = 0.0006 (meters) chosen for Lx = 1.1cm, Pe = 7.23E6
elif flow_rate == 30.0:
    Ly = Constant(0.0004) #domain height (y-length) = 0.0004 (meters) chosen for Lx = 2.2cm, Pe = 7.23E7
Lb = Constant(0.0011) #outlet section x-length (meters)

nx = 940 #number of cells in x-direction
ny = 27 #number of cells in y-direction
ny_refined_region = 5 #inner refinement level: number of cells in y-direction in the refined region (adjacent to catalyst)
Ly_refined_region = 1e-6 #inner refinement level: length in y-direction of the refined region (adjacent to catalyst)
mesh = RectangleBoundaryLayerMesh(nx, ny, La.dat.data + Lx.dat.data + Lb.dat.data, Ly.dat.data, ny_refined_region, Ly_refined_region, boundary=(3,)) #construct mesh, using echemfem

Vc = mesh.coordinates.function_space() #obtain function space of mesh coordinates
x, y = SpatialCoordinate(mesh) #obtain coordinates of mesh


####-------------outer refinement level: refine elements in the 1/3 of y-domain closest to catalyst -------------------
ratio = 0.75 # ratio of elements in the first 1/3
new_y = conditional(le(y, 1e-6), y, conditional(lt(y, ratio*Ly), y * 1/3 / ratio, (y-ratio*Ly) * 2/3 / (1-ratio) + 1/3 * Ly))
f = Function(Vc).interpolate(as_vector([x, new_y]))
with stop_annotating():
    mesh.coordinates.assign(f)

print('inlet section La (meters): ',La.dat.data,', electrode length Lx (meters): ',Lx.dat.data,', outlet section Lb (meters): ',Lb.dat.data,', domain height Ly (meters): ',Ly.dat.data)
print('mesh parameters; nx: ',nx,', ny: ',ny,', ny_refined_region: ',ny_refined_region,', Ly_refined_region: ',Ly_refined_region,', ratio of mesh elements in first 1/3: ',ratio)

##-----mesh/domain parameters end----------------------------------------------



class CarbonateSolver(EchemSolver):
    def __init__(self, rho_input = None):
        #list of references
        """
        Bicarbonate flow reactor setup given in:
        Lin, T.Y., Baker, S.E., Duoss, E.B. and Beck, V.A., 2021. Analysis of
        the Reactive CO2 Surface Flux in Electrocatalytic Aqueous Flow
        Reactors. Industrial & Engineering Chemistry Research, 60(31),
        pp.11824-11833.

        Ag Tafel parameter values taken from: 
        Corpus, K. R. M., Bui, J. C., Limaye, A. M., Pant, L. M., Manthiram, K., 
        Weber, A. Z., and Bell, A. T., 2023. Coupling covariance matrix adaptation 
        with continuum modeling for determination of kinetic parameters associated 
        with electrochemical CO2 reduction. Joule, 7, pp.1289-1307.

        Cu Tafel parameter values taken from:
        Li, J., Chang, X., Zhang, H., Malkani, A. S., Cheng, M. J., Xu, B., 
        and Lu, Q., 2021. Electrokinetic and in situ spectroscopic investigations 
        of CO electrochemical reduction on copper. Nature Communications, 12(1), pp.3264.

        Diffusivities values taken from:
        Weng, L. C., Bell, A. T., & Weber, A. Z., 2018. Modeling gas-diffusion 
        electrodes for CO2 reduction. Physical Chemistry Chemical Physics, 20(25), 
        pp. 16973-16984.
        and from: 
        Cussler, E. L. Diffusion: mass transfer in fluid systems; Cambridge University Press, 2009.

        The mesh has been taken from that used in (and then slightly modified for this smaller domain size):
        Govindarajan, N., Lin, T. Y., Roy, T., Hahn, C., & Varley, J. B., 2023. Coupling 
        Microkinetics with Continuum Transport Models to Understand Electrochemical 
        CO2 Reduction in Flow Reactors. PRX Energy, 2(3), pp.033010.

        01/11/2024: add in 1-defect optimization capabilities, as analogously implemented in tworxn_GMSH_mesh_FOR_BASH_SCRIPT_WITH_OPTIMIZATION_ATTEMPT_1.py
        02/2024: attempting to add in multi-defect capability (as generalization of bicarb_Ag_Cu_2D_HALF_HALF_with_Mesh_File_WITH_OPTIMIZATION_ATTEMPT_1.py)
        09/2024: removed unnecessary species that don't need to be solved for (C2+, CH4, H2), but keep original bulk concentration formulation
        09/2024: using original bulk reaction system (in bulk_reaction()) but now using updated reaction constants and updated fsolve() method to solve for bulk concentrations
        """
        

        ####-------------------begin set up functions for Ag, Cu, and defect local BCs


        #2 options: rho_input isn't/is passed as argument input to CarbonateSolver object
        if (rho_input == None):
            self.rho_list = [ Constant( 0. )  for i in range (n_sections-1)] #use this option to initialize all rho_i = 0
            #self.rho_list = [Constant( Lx/n_sections ) for i in range (n_sections-1)] #use this option to initialize all rho_i = Lx/n_sections
        else:
            self.rho_list = rho_input

        ####-----------------------------section length formulation from rho_list
        #get the lengths of each section from rho_list
        length_list = [Constant(0.) for i in range(n_sections)]
        for index in range(n_sections):
            temp1 = Constant(1.)
            temp2 = Constant(1.)
            for index_temp1 in range(n_sections - index - 1):
                temp1 = temp1 * self.rho_list[index_temp1]
            if index != 0:
                temp2 = Constant(1.) - self.rho_list[n_sections - 1 - index]
            length_list[index] = temp1 * temp2 * Constant(Lx)
        print('printing length_list in CarbonateSolver class: ')
        for index in range(len(length_list)):
            print(Constant(length_list[index]).dat.data)

        #and then recompute (i.e. reassign elements of) bounds_vector based on length_list from rho_list
        self.bounds_vector = []
        self.bounds_vector.append(Constant(0.))
        for index in range(1,n_sections):
            temp = self.bounds_vector[index-1] + length_list[index-1]
            self.bounds_vector.append(temp)
        self.bounds_vector.append(Constant(Lx))
        print('printing bounds_vector in CarbonateSolver class')
        for index in range(n_sections+1):
            print(Constant(self.bounds_vector[index]).dat.data)

        #check that sum of elements of length_list equals Lx
        print('check that sum of elements of length_list equals Lx')
        sum_ll = Constant(0.0)
        for index in range(n_sections):
            sum_ll = sum_ll + length_list[index]
        print('sum of length_list elements - Lx = ',Constant(sum_ll).dat.data,' - ',Constant(self.bounds_vector[n_sections]).dat.data,' = (should equal 0) ',Constant(sum_ll - self.bounds_vector[n_sections]).dat.data)

    
        #regularization (sharpening) factor for tanh; larger values correspond to weaker regularization and a steeper tanh profile
        sharpening_factor = 57.5E3

        
        BC_local_Ag = x * 0.0
        BC_local_Cu = x * 0.0
        ####------------------initialize Ag BC sections with tanh
        for index in range(0,n_sections,2):
            BC_local_Ag += 0.5 * (  tanh(sharpening_factor * (x - self.bounds_vector[index] - Constant(La)) )  -  tanh(sharpening_factor * (x - self.bounds_vector[index+1] - Constant(La)) ) )
        ##------------------------------------------------------------------
        ####-----------------initialize Cu BC sections with tanh
        for index in range(0,n_sections,2):
            BC_local_Cu += 0.5 * (  tanh(sharpening_factor * (x - self.bounds_vector[index+1] - Constant(La)) )  -  tanh(sharpening_factor * (x - self.bounds_vector[index+2] - Constant(La)) ) )
        ##------------------------------------------------------------------
        
        ##-----------------------------end, section length formulation from rho_list


        ##-------------------end set up functions for Ag, Cu, and defect local BCs


        ####-----------bulk concentrations: values of constants + initial guesses for concentrations
        #note that these values of K, kr, kf are slightly different from Lin et al ACS I&EC 2021; 
        # The values reported don't solve this system with very high accuracy, which is
        # problematic since this causes a discontinuity at the boundary condition.
        # Instead, we fix CO2 and K+ concentrations and solve for the others using law
        # of mass action and electroneutrality. Note that two of the reactions are
        # linearly dependent on the others (see above how K3 and K4 are obtained from
        # the others).
        KW = (1E-14)*pow(1000,2)    #(mol/m^3)^2
        K1 = pow(10,-6.37)*1000     #(mol/m^3)
        K2 = pow(10,-10.32)*1000    #(mol/m^3)
        K3 = K1/KW                  #(mol/m^3)^(-2)
        K4 = K2/KW                  #(mol/m^3)^(-1)
        k1f = 8.42e3 / 1000         #(m^3/mol/s)
        k5f = 2.3e10 / 1000         #(m^3/mol/s) ; Note: typo in Schulz et al. Marine chemistry 2006.
        k1r = k1f/K3                #mol/(m^3*s)
        k2f = 3.71e-2               #1/s
        k2r = k2f/K1                #(m^3/mol/s)
        k3r = 6.0e9 / 1000          #(m^3/mol/s)
        k3f = k3r/K4                #1/s
        k4r = 59.44                 #1/s
        k4f = k4r/K2                #(m^3/mol/s)
        k5r = KW*k5f                #(mol/m^3/s) #Note: KW = k5

        C_1_inf = 0.034*1000 # mol/m^3
        C_K =  0.5*1000 # mol/m^3, for 500mM KHCO3 buffer solution
        Keq = k1f*k3f/(k1r*k3r)

        #solve for initial guesses for concentrations
        C_3_inf = -C_1_inf*Keq/4 + C_1_inf*Keq/4*sqrt(1+8*C_K/(C_1_inf*Keq))
        C_4_inf = (C_K - C_3_inf)/2
        C_2_inf = C_3_inf/C_1_inf*k1r/k1f
        C_5_inf = (k5r/k5f)/C_2_inf

        C_CO2_bulk = C_1_inf
        C_OH_bulk = C_2_inf
        C_HCO3_bulk = C_3_inf
        C_CO32_bulk = C_4_inf
        C_H_bulk = C_5_inf
        C_K_bulk = C_K

        #note that CO is not present in the initial solution

        ####--------------------Make sure that electroneutrality holds by manually adjusting K ion concentration
        netcharge = C_K_bulk+ C_H_bulk -2.*C_CO32_bulk - C_HCO3_bulk - C_OH_bulk
        print("Total charge (init) is:",netcharge )
        C_K_bulk = C_K_bulk - netcharge
        netcharge = C_K_bulk+ C_H_bulk -2.*C_CO32_bulk - C_HCO3_bulk - C_OH_bulk
        print("Total charge (init) is:",netcharge )
        ##--------------------

        print('CO2 bulk init',C_CO2_bulk)
        print('OH bulk init',C_OH_bulk)
        print('HCO3_bulk init',C_HCO3_bulk)
        print('CO32_bulk init',C_CO32_bulk)
        print('H_bulk init',C_H_bulk)
        print('CO_bulk init',0.)
        print('K_bulk init',C_K_bulk)

        
        print('K1',K1)
        print('K2',K2)
        print('K3',K3)
        print('K4',K4)
        print('Kw',KW)

        ##-----------end, bulk concentrations: values of constants + initial guesses for concentrations

        ####-----------calculate bulk values
        def reaction_system(u): #subfunction used for solving the bulk reaction system; Note: taken from EchemFEM examples/catalyst_layer_Cu_full.py
            C_CO2 = C_CO2_bulk
            C_K = C_K_bulk
            C_OH = u[0]
            C_HCO3 = u[1]
            C_CO3 = u[2]
            C_H = u[3]
            r3 = 1. - C_HCO3 / (K3 * C_OH * C_CO2)
            r4 = 1. - C_CO3 / (K4 * C_HCO3 * C_OH)
            rw = 1. - (C_OH * C_H)/KW
            electro = C_K + C_H - C_OH - C_HCO3 - 2 * C_CO3
            return [r3, r4, rw, electro]

        Ci = [C_OH_bulk, C_HCO3_bulk, C_CO32_bulk, C_OH_bulk]
        Copt = fsolve(reaction_system, Ci, xtol=1e-12, maxfev=1000, diag=None)
        C_OH_bulk = Copt[0]
        C_HCO3_bulk = Copt[1]
        C_CO32_bulk = Copt[2]
        C_H_bulk = Copt[3]

        netcharge = C_K_bulk+ C_H_bulk -2.*C_CO32_bulk - C_HCO3_bulk - C_OH_bulk
        print("Total charge is:",netcharge )

        print('CO2 bulk ',C_CO2_bulk)
        print('OH bulk',C_OH_bulk)
        print('HCO3_bulk',C_HCO3_bulk)
        print('CO32_bulk',C_CO32_bulk)
        print('H_bulk',C_H_bulk)
        print('CO_bulk',0.)
        print('K_bulk',C_K_bulk)

        ##-----------end, calculate bulk values



        #define reaction set for bulk reactions
        def bulk_reaction(y):
            yCO2=y[0];
            yOH=y[1];
            yHCO3=y[2];
            yCO3=y[3];
            yH=y[4];
            
            
            dCO2 = -(k1f)   *yCO2*yOH \
                    +(k1r)    *yHCO3 \
                    -(k2f)  *yCO2 \
                    +(k2r)   *yHCO3*yH

            dOH = -(k1f)       *yCO2*yOH \
                           +(k1r)        *yHCO3 \
                           +(k3f)        *yCO3 \
                           -(k3r)       *yOH*yHCO3\
                           -(k5f)      *yOH*yH\
                           +(k5r)

            dHCO3 = (k1f)        *yCO2*yOH\
                           -(k1r)       *yHCO3\
                           +(k3f)        *yCO3\
                           -(k3r)       *yOH*yHCO3\
                           +(k2f)       *yCO2 \
                           -(k2r)      *yHCO3*yH \
                           +(k4f)       *yCO3*yH\
                           -(k4r)      *yHCO3

            dCO3 = -(k3f) *yCO3 \
                           +(k3r)  *yOH*yHCO3\
                           -(k4f)*yCO3*yH\
                           +(k4r) *yHCO3

            dH = (k2f)   *yCO2 \
                           -(k2r)  *yHCO3*yH \
                           -(k4f)  *yCO3*yH\
                           +(k4r)   *yHCO3\
                           -(k5f)  *yOH*yH\
                           +(k5r)
               
            #return [dCO2, dOH, dHCO3, dCO3, dH, 0., 0., 0., 0., 0., 0.] #return list used for the full reaction system, which includes H2, C2H4, C2H6O, CH4
            return [dCO2, dOH, dHCO3, dCO3, dH, 0., 0.] # in the optimization script, H2, C2H4, C2H6O, CH4 are omitted and so this shortened return list is used


        conc_params = []

        #diffusivities from Weng et al. Physical Chemistry Chemical Physics 2018.
        #and from Cussler, E. L. Diffusion: mass transfer in fluid systems; Cambridge University Press, 2009.
        conc_params.append({"name": "CO2",
                            "diffusion coefficient": 1.91E-9,  # m^2/s
                            "bulk": C_CO2_bulk,  # mol/m3
                            "z": 0,
                            })

        conc_params.append({"name": "OH",
                            "diffusion coefficient": 5.29E-9,  # m^2/s
                            "bulk": C_OH_bulk,  # mol/m3
                            "z": -1,
                            })

        conc_params.append({"name": "HCO3",
                            "diffusion coefficient": 1.185E-9,  # m^2/s
                            "bulk": C_HCO3_bulk,  # mol/m3
                            "z": -1,
                            })

        conc_params.append({"name": "CO3",
                            "diffusion coefficient": .92E-9,  # m^2/s
                            "bulk": C_CO32_bulk,  # mol/m3
                            "z": -2,
                            })

        conc_params.append({"name": "H",
                            "diffusion coefficient": 9.311E-9,  # m^2/s
                            "bulk": C_H_bulk,  # mol/m3
                            "z": 1,
                            })

        ####-----------H2 not included in the optimization reaction set
        # conc_params.append({"name": "H2", 
        #                     "diffusion coefficient": 4.5E-9, #1.96E-9,  # m^2/s
        #                     "bulk": 0.0,  # mol/m3
        #                     "z": 0,
        #                     })
        ##-----------

        conc_params.append({"name": "CO",
                            "diffusion coefficient": 2.03E-9, #1.96E-9,  # m^2/s
                            "bulk": 0.0,  # mol/m3
                            "z": 0,
                            })

        ####------------------C2H4, C2H6O, CH4 not included in the optimization reaction set
        # conc_params.append({"name": "C2H4",
        #                     "diffusion coefficient": 1.87E-9,  # m^2/s
        #                     "bulk": 0.,  # mol/m3
        #                     "z": 0,
        #                     })

        # conc_params.append({"name": "C2H6O",
        #                     "diffusion coefficient": 0.84E-9,  # m^2/s
        #                     "bulk": 0.,  # mol/m3
        #                     "z": 0, 
        #                     })

        # conc_params.append({"name": "CH4",
        #                     "diffusion coefficient": 1.49E-9,  # m^2/s
        #                     "bulk": 0.,  # mol/m3
        #                     "z": 0, 
        #                     })
        ##-----------

        conc_params.append({"name": "K",
                    "diffusion coefficient": 1.957E-9, #1.96E-9,  # m^2/s
                    "bulk": C_K_bulk,  # mol/m3
                    "z": 1,
                    })

        physical_params = {"flow": ["advection", "diffusion", "migration", "electroneutrality"],
                   "F": F,  # C/mol
                   "R": R,  # J/K/mol
                   "T": T,  # K
                   "U_app": Vcell, # V vs. SHE
                   "bulk reaction": bulk_reaction,
                   }

        #Ag CO reaction from Tafel
        def reaction_CO(u):
            CCO2 = u[0]
            COH = u[1]
            CHCO3 = u[2]
            CCO3 = u[3]
            CH = u[4]
            #(omitted) #CH2 = u[5]
            CCO = u[5]#(reassigned index) #u[6]
            CK = -u[4] + 2*u[3] + u[2] + u[1]
            #(omitted) #CC2H4 = u[7]
            #(omitted) #CC2H6O = u[8]
            #(omitted) #CCH4 = u[9]
            Phi2 = u[6]#(reassigned index) #u[10]
            Phi1 = physical_params["U_app"]
            UCO = U0_CO

            ####-------calculate activities--------------------
            #charges of species
            z_CO2 = 0.
            z_OH = -1.
            z_HCO3 = -1.
            z_CO3 = -2.
            z_H = 1.
            z_H2 = 0.
            z_CO = 0.
            z_K = 1.

            #local ionic strength to calculate a_OH and a_H
            I =  0.5 * ( CCO2*pow(z_CO2,2) + COH*pow(z_OH,2) + CHCO3*pow(z_HCO3,2) + CCO3*pow(z_CO3,2) + CH*pow(z_H,2) + CK*pow(z_K,2)) #local ionic strength
            I = I/1000. #convert from mol/m^3 to molar
            f_OH = pow(10.,-0.51 * pow(z_OH,2) * (pow(I,0.5)/(1. + pow(I,0.5)) - 0.3 * I ) )
            f_H = pow(10.,-0.51 * pow(z_H,2) * (pow(I,0.5)/(1. + pow(I,0.5)) - 0.3 * I ) )

            #activities and pH
            a_OH = f_OH * COH / cref_Ag
            a_OH_bulk = C_OH_bulk / cref_Ag

            a_H = f_H * CH / cref_Ag
            pH = -ln(a_H)/ln(10)

            f_CO2 = exp(COH*(h_s_OH+h_g_CO2) + CHCO3*(h_s_HCO3+h_g_CO2) + CCO3*(h_s_CO3+h_g_CO2) + CH*(h_s_H+h_g_CO2) + CK*(h_s_K+h_g_CO2))
            a_CO2 = f_CO2 * CCO2 / cref_Ag
            a_CO2_bulk = C_CO2_bulk / cref_Ag
            ##--------calculate activities            

            eta_CO = Phi1 - Phi2 - (UCO - ((2.303*R*T)/F)*(pH) + (R*T)/(2*F)*(ln(a_CO2))) # reaction overpotential (V vs. SHE)
            iCO = i0_CO * (a_CO2 / a_CO2_bulk)**(-gamma_CO2_CO) * (a_OH/a_OH_bulk)**(gamma_OH_CO) * exp(-((alpha_c_CO * F) / (R * T)) * eta_CO) # current density towards CO, A/m^2
            return iCO * BC_local_Ag


        #Ag H2 reaction from Tafel
        def reaction_H2_Ag(u):
            CCO2 = u[0]
            COH = u[1]
            CHCO3 = u[2]
            CCO3 = u[3]
            CH = u[4]
            #(omitted) #CH2 = u[5]
            CCO = u[5]#(reassigned index) #u[6]
            CK = -u[4] + 2*u[3] + u[2] + u[1]
            #(omitted) #CC2H4 = u[7]
            #(omitted) #CC2H6O = u[8]
            #(omitted) #CCH4 = u[9]
            Phi2 = u[6]#(reassigned index) #u[10]
            Phi1 = physical_params["U_app"]

            ####---------calculate activities------------------
            #charges of species
            z_CO2 = 0.
            z_OH = -1.
            z_HCO3 = -1.
            z_CO3 = -2.
            z_H = 1.
            z_H2 = 0.
            z_CO = 0.
            z_K = 1.

            #local ionic strength to calculate a_OH and a_H
            I =  0.5 * ( CCO2*pow(z_CO2,2) + COH*pow(z_OH,2) + CHCO3*pow(z_HCO3,2) + CCO3*pow(z_CO3,2) + CH*pow(z_H,2) + CK*pow(z_K,2)) #local ionic strength
            I = I/1000. #convert from mol/m^3 to molar
            f_OH = pow(10.,-0.51 * pow(z_OH,2) * (pow(I,0.5)/(1. + pow(I,0.5)) - 0.3 * I ) )
            f_H = pow(10.,-0.51 * pow(z_H,2) * (pow(I,0.5)/(1. + pow(I,0.5)) - 0.3 * I ) )

            #activities and pH
            a_OH = f_OH * COH / cref_Ag
            a_OH_bulk = C_OH_bulk / cref_Ag

            a_H = f_H * CH / cref_Ag
            pH = -ln(a_H)/ln(10)
            ##----------calculate activities------------------------------

            eta_H2_Ag = Phi1 - Phi2 - (U0_H2_Ag - ((2.303*R*T)/F)*(pH)) # reaction overpotential (V vs. SHE)
            iH2_Ag = i0_H2_Ag * (a_OH/a_OH_bulk)**(gamma_OH_H2) * exp(-((alpha_c_H2_Ag * F) / (R * T)) * eta_H2_Ag) # current density towards H2, A/m^2
            return iH2_Ag * BC_local_Ag



        #Cu C2H4 reaction from Tafel
        def reaction_C2H4(u):
            C_CO = u[5]#(reassigned index) #u[6]
            Phi2 = u[6]#(reassigned index) #u[10]
            Phi1 = physical_params["U_app"]

            eta_C2H4 = Phi1 - Phi2 # reaction overpotential (V vs. SHE)
            iC2H4 = i0_C2H4 * (C_CO/cref) * exp(-((alpha_c_C2H4 * F) / (R * T)) * eta_C2H4) # current density towards C2H4, A/m^2
            return iC2H4 * BC_local_Cu

        #Cu C2H6O reaction from Tafel
        def reaction_C2H6O(u):
            C_CO = u[5]#(reassigned index) #u[6]
            Phi2 = u[6]#(reassigned index) #u[10]
            Phi1 = physical_params["U_app"]

            eta_C2H6O = Phi1 - Phi2 # reaction overpotential (V vs. SHE) 
            iC2H6O = i0_C2H6O * (C_CO/cref) * exp(-((alpha_c_C2H6O * F) / (R * T)) * eta_C2H6O) # current density towards C2H6O, A/m^2
            return iC2H6O * BC_local_Cu

        #Cu H2 reaction from Tafel
        def reaction_H2_Cu(u):
            Phi2 = u[6]#(reassigned index) #u[10]
            Phi1 = physical_params["U_app"]
    
            eta_H2 = Phi1 - Phi2 # reaction overpotential (V vs. SHE)
            iH2_Cu = i0_H2_Cu * exp(-((alpha_c_H2_Cu * F) / (R * T)) * eta_H2) # current density towards H2, A/m^2
            return iH2_Cu * BC_local_Cu

        #Cu CH4 reaction from Tafel
        def reaction_CH4(u):
            C_CO = u[5]#(reassigned index) #u[6]
            Phi2 = u[6]#(reassigned index) #u[10]
            Phi1 = physical_params["U_app"]
    
            eta_CH4 = Phi1 - Phi2 # reaction overpotential (V vs. SHE)
            iCH4 = i0_CH4 * (C_CO/cref) * exp(-((alpha_c_CH4 * F) / (R * T)) * eta_CH4) # current density towards CH4, A/m^2
            return iCH4 * BC_local_Cu


        #add surface reactions to echem_params
        #H2, C2H4, C2H6O, CH4 omitted
        echem_params = []

        echem_params.append({"reaction": reaction_CO,
                             "electrons": 2,
                             "stoichiometry": {"CO2": -1,
                                               "OH": 2,
                                               "CO": 1}, # product
                             "boundary": "catalyst",
                             })

        echem_params.append({"reaction": reaction_H2_Ag,
                             "electrons": 2,
                             "stoichiometry": {"OH": 2},
                                               #"H2": 1}, # product
                             "boundary": "catalyst",
                             })

        echem_params.append({"reaction": reaction_C2H4,
                             "electrons": 8,
                             "stoichiometry": {"CO": -2,
                                               #"C2H4": 1,
                                               "OH": 8}, # product
                             "boundary": "catalyst",
                             })

        echem_params.append({"reaction": reaction_C2H6O,
                             "electrons": 8,
                             "stoichiometry": {"CO": -2,
                                               #"C2H6O": 1,
                                               "OH": 8}, # product
                             "boundary": "catalyst",
                             })
        echem_params.append({"reaction": reaction_H2_Cu,
                             "electrons": 2,
                             "stoichiometry": {"OH": 2},# product
                                               #"H2": 1}, 
                             "boundary": "catalyst",
                             })
        echem_params.append({"reaction": reaction_CH4,
                             "electrons": 6,
                             "stoichiometry": {"CO": -1,
                                               #"CH4": 1,
                                               "OH": 6}, # product
                             "boundary": "catalyst",
                             })

        super().__init__(conc_params, physical_params, mesh, echem_params=echem_params, family="DG")
        #super().__init__(conc_params, physical_params, mesh, echem_params=echem_params, family="CG")#, SUPG = True) #other options while solving

    #set boundary conditions
    def set_boundary_markers(self):
        self.boundary_markers = {"inlet": (1),
                                 "bulk dirichlet": (4),#this line is not strictly necessary
                                 "outlet": (2,),
                                 "catalyst": (3,),
                                 "bulk": (4,), 
                                 }


    #set electrolyte shear flow based on flow rate input
    def set_velocity(self):
        total_domain_length_cm = 1.1 #in cm, #haven't been able to call Lx as a double value, so this value is hard-coded in and needs to be updated if Lx is changed
        print('total_domain_length, cm (printed in set_velocity())')
        print(total_domain_length_cm)

        _, y = SpatialCoordinate(self.mesh)
        if flow_rate == 3.0:
            self.vel = as_vector([7.23E1 /(total_domain_length_cm*total_domain_length_cm) * 1.91E0*y,Constant(0)]) # m/s #for Pe = 7.23E6 (for 3ml/min)
        elif flow_rate == 30.0:
            self.vel = as_vector([7.23E2 /(total_domain_length_cm*total_domain_length_cm) * 1.91E0*y,Constant(0)]) # m/s #for Pe = 7.23E7 (for 30ml/min)


####-------------OPTIMIZATION: define __main__ section
if __name__ == "__main__":
    ####----------------initialize each element of rho_list control list; this becomes a list (rho_0, rho_1, ..., rho_N-2)
    rho_init_list = [Constant(0.)]

    #uncomment these lines with the choice of rho_i = 1/n_sections or rho_i = 0.5 if desire is to initialize this way
    #--------------------------------------------------------------------
    #for index in range(n_sections - 1):
        #rho_init_list.append(  Constant( 1./n_sections )  )
        #rho_init_list.append(  Constant( 0.5 )  )
    #rho_init_list.pop(0)
    #--------------------------------------------------------------------

    #these lines initializes such that all sections have same length ( = Lx/n_sections)
    #--------------------------------------------------------------------
    rho_for_const_sec_length_list = [Constant(0.)]#calculate rho_i such that each initial section length is the same ( = Lx/n_sections)
    rho_for_const_sec_length_list.append( Constant(1. - 1./n_sections) )
    rho_for_const_sec_length_list.pop(0)
    for index in range(1, n_sections - 1, 1):
        temp = Constant(1.)
        for index_inner in range(0,index,1): #loop over previous rho_i values up to index-1 in order to form the denominator
            temp = temp * rho_for_const_sec_length_list[index_inner]
        rho_for_const_sec_length_list.append( Constant( 1. - (1./n_sections)/temp ) )
    rho_init_list = rho_for_const_sec_length_list 
    #--------------------------------------------------------------------

    print('printing control list rho_list initialized before optimization, at beginning of main: ')
    for index in range(len(rho_init_list)):
        print(rho_init_list[index].dat.data)

    ##----------------end initialize each element of rho_list control list

    ####----------------set up solver; initial U_app value should be low (less negative) enough that no continuation (voltage loop) is needed to reach this initial value --------------------------------------------------------
    solver = CarbonateSolver(rho_init_list)
    solver.U_app.assign(Vcell)
    #if need to increase max # iterations, use custom_solver options found in EchemFEM echemfem/solver.py in init_solver_parameters()
    solver.setup_solver()
    u_control = Control(solver.u) #this extracts the solution field
    solver.solve() #this is a forward solve
    ##--------------------------------------------------------


    #compute and print electrode length, as confirmation, before optimization
    total_electrode_length_for_opt = Constant(0.)
    for index in range(0,n_sections,2):
        total_electrode_length_for_opt = total_electrode_length_for_opt + solver.bounds_vector[index+1] - solver.bounds_vector[index] + solver.bounds_vector[index+2] - solver.bounds_vector[index+1]
    print('total_electrode_length before optimization: ', Constant(total_electrode_length_for_opt).dat.data )



    n = FacetNormal(solver.mesh) #normal vector
    #cCO2, cOH, cHCO3, cCO3, cH, cH2, cCO, cC2H4, cC2H6O, cCH4, phi2 = solver.u.split() #includes all species; not used for this optimization solve
    cCO2, cOH, cHCO3, cCO3, cH, cCO, phi2 = solver.u.split() #omits H2, C2H4, C2H6O, CH4

    ####--------------------------this is for writing optimization iterations to files; not used for now
    def eval_cb(j, boundary_d_for_output):
        pass
    ##---------------------------------------------------------------------------------------------


    ####-----------continuation loop to reach desired U_app voltage values
    Vlist = np.append(np.linspace(-1.35,-1.55,num=21),np.linspace(-1.555,-1.75,num=40))
    Vlist = Vlist[Vlist>=-V_final]
    if -V_final not in Vlist: Vlist = np.append(Vlist,-V_final)
    for Vs in Vlist:
        solver.U_app.assign(Vs)
        print("V = %d mV" % np.rint(Vs * 1000))
        solver.solve()
    ##---------------------------------------------------------

    ####--------------compute species specific current density values at wall; as needed for functional, reduced functional, control
    #Note: these are computed with total domain length (La + Lx + Lb) as denominator; they will be normalized to correct active length (Lx) in denominator for outputting later on
    iCO_avg = assemble(solver.echem_params[0]["reaction"](solver.u.subfunctions)*solver.ds(3))/assemble(Constant(1) * solver.ds(3, domain=solver.mesh))
    iH2_Ag_avg = assemble(solver.echem_params[1]["reaction"](solver.u.subfunctions)*solver.ds(3))/assemble(Constant(1) * solver.ds(3, domain=solver.mesh))
    iC2H4_avg = assemble(solver.echem_params[2]["reaction"](solver.u.subfunctions)*solver.ds(3))/assemble(Constant(1) * solver.ds(3, domain=solver.mesh))
    iC2H6O_avg = assemble(solver.echem_params[3]["reaction"](solver.u.subfunctions)*solver.ds(3))/assemble(Constant(1) * solver.ds(3, domain=solver.mesh))
    iH2_Cu_avg = assemble(solver.echem_params[4]["reaction"](solver.u.subfunctions)*solver.ds(3))/assemble(Constant(1) * solver.ds(3, domain=solver.mesh))
    iCH4_avg = assemble(solver.echem_params[5]["reaction"](solver.u.subfunctions)*solver.ds(3))/assemble(Constant(1) * solver.ds(3, domain=solver.mesh))

    denom_FE = iCO_avg + iH2_Ag_avg + iC2H4_avg + iC2H6O_avg + iH2_Cu_avg + iCH4_avg
    FE_CO = iCO_avg / denom_FE
    FE_H2 = (iH2_Ag_avg + iH2_Cu_avg) / denom_FE
    FE_C2H4 = iC2H4_avg / denom_FE
    FE_C2H6O = iC2H6O_avg / denom_FE
    FE_CH4 = iCH4_avg / denom_FE
    ##--------------end, compute species specific current density values at wall; as needed for functional, reduced functional, control


    ##create control list version of rho_list (rho_list_control), to be used in ReducedFunctional for optimization within each iteration
    rho_list_control = [Control (Constant(0.) ) ]
    for index in range(len(solver.rho_list)):
        rho_list_control.append( Control(solver.rho_list[index]) )
    rho_list_control.pop(0)

    #from rho_list_control, extract and then print the lengths of each section (to be printed within each iteration)
    def deriv_cb(j, dj, gamma):
        with stop_annotating():
            for idx, rho in enumerate(rho_list_control):
                print("Control value ", idx, " = ", rho.tape_value().values()[0])
            #get the lengths of each section from rho_list
            length_list = [Constant(0.) for i in range(n_sections)]
            for index in range(n_sections):
                temp1 = Constant(1.)
                temp2 = Constant(1.)
                for index_temp1 in range(n_sections - index - 1):
                    temp1 = temp1 * rho_list_control[index_temp1].tape_value()
                if index != 0:
                    temp2 = Constant(1.) - rho_list_control[n_sections - 1 - index].tape_value()
                length_list[index] = temp1 * temp2 * Constant(Lx)
            print('printing length_list in CarbonateSolver class: ')
            for index in range(len(length_list)):
                print(Constant(length_list[index]).dat.data)
    
    #define functional and reduced functional
    if objective == 'current':
        J = -iC2H4_avg # maximize current
    elif objective == "FE":
        J = -FE_C2H4 # Faradaic efficiency
    m =  rho_list_control #control_list
    Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb, derivative_cb_post=deriv_cb)

    ## Bound constraints (ensure that 0<=rho_j<=1)
    lb = 0.0
    ub = 1.0

    #print progress bar to visualize optimization solve progress
    get_working_tape().progress_bar = ProgressBar

    #call scipy.minimize, to solve the optimization problem given the setup
    f_opt = minimize(Jhat, method = 'L-BFGS-B',bounds=(lb,ub), options={'disp': True})
    print('printing f_opt with disp True')
    print(f_opt)


    #printed final optimized values of rho
    print('f_opt.dat.data after solve: ')
    if n_sections > 2:
        for index in range(len(f_opt)):
            print( f_opt[index].dat.data )
    else:
        print( f_opt.dat.data )
        ###----------redefine f_opt to be a list
        f_opt_list = [Constant(0.)]
        f_opt_list.append(Constant(f_opt))
        f_opt_list.pop(0)
        f_opt = f_opt_list

    #print value of initial and final reduced functional
    print("Jhat(g) = %.8g\nJhat(g_opt) = %.8g" % (Jhat(rho_init_list) * (La + Lx + Lb) / (Lx), Jhat(f_opt) * (La + Lx + Lb) / (Lx) ) )


    ####------------------ re-solve with optimized f_opt to write results (as paraview-readable files) and also print final integrated current density values
    #Note: sometimes, the forward solve here may not converge; this is okay, as the goal of the optimization script is to output the converged optimized section lengths, these can then be used as input for the final forward script file
    #re-add-in U_app continuation loop is implemented, starting from lower (less-negative) value
    Vlist_final = np.append(np.linspace(-0.5,-1.3,num=9),Vlist)
    for Vs in Vlist_final:
        for idx, rho in enumerate(f_opt):
            solver.rho_list[idx].assign(rho)#Note: rho is equivalent to f_opt[idx]
        
        solver.U_app.assign(Vs)
        print("for optimized solve, V = %d mV" % np.rint(Vs * 1000))
        
        #solver.u.assign(u_control.tape_value())#assign final state as initial guess for forward solve, to help with convergence; this is an option that can be used in place of the line below
        solver.solve()

    #cCO2, cOH, cHCO3, cCO3, cH, cH2, cCO, cC2H4, cC2H6O, cCH4, phi2 = solver.u.subfunctions #includes all species; not used for this optimization solve
    cCO2, cOH, cHCO3, cCO3, cH, cCO, phi2 = solver.u.subfunctions #omits H2, C2H4, C2H6O, CH4

    ####-------------------print optimized current density values as computed by this optimization script
    print('printing domain-averaged i_CO')
    iCO_avg = assemble(solver.echem_params[0]["reaction"](solver.u.subfunctions)*solver.ds(3))/assemble(Constant(1) * solver.ds(3, domain=solver.mesh))
    print(iCO_avg * (La.dat.data + Lx.dat.data + Lb.dat.data) / (Lx.dat.data))

    print('printing domain-averaged i_H2_Ag')
    iH2_Ag_avg = assemble(solver.echem_params[1]["reaction"](solver.u.subfunctions)*solver.ds(3))/assemble(Constant(1) * solver.ds(3, domain=solver.mesh))
    print(iH2_Ag_avg * (La.dat.data + Lx.dat.data + Lb.dat.data) / (Lx.dat.data))

    print('printing domain-averaged i_C2H4')
    iC2H4_avg = assemble(solver.echem_params[2]["reaction"](solver.u.subfunctions)*solver.ds(3))/assemble(Constant(1) * solver.ds(3, domain=solver.mesh))
    print(iC2H4_avg * (La.dat.data + Lx.dat.data + Lb.dat.data) / (Lx.dat.data))

    print('printing domain-averaged i_C2H6O')
    iC2H6O_avg = assemble(solver.echem_params[3]["reaction"](solver.u.subfunctions)*solver.ds(3))/assemble(Constant(1) * solver.ds(3, domain=solver.mesh))
    print(iC2H6O_avg * (La.dat.data + Lx.dat.data + Lb.dat.data) / (Lx.dat.data))

    print('printing domain-averaged i_H2_Cu')
    iH2_Cu_avg = assemble(solver.echem_params[4]["reaction"](solver.u.subfunctions)*solver.ds(3))/assemble(Constant(1) * solver.ds(3, domain=solver.mesh))
    print(iH2_Cu_avg * (La.dat.data + Lx.dat.data + Lb.dat.data) / (Lx.dat.data))

    print('printing domain-averaged i_CH4')
    iCH4_avg = assemble(solver.echem_params[5]["reaction"](solver.u.subfunctions)*solver.ds(3))/assemble(Constant(1) * solver.ds(3, domain=solver.mesh))
    print(iCH4_avg * (La.dat.data + Lx.dat.data + Lb.dat.data) / (Lx.dat.data))

    ##-------------------------------------------------------
    ##-------------------------------------------------------

##-------------end, OPTIMIZATION: define __main__ section
