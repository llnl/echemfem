Examples
========

Here is a brief overview of the examples in `echemfem/examples <https://github.com/LLNL/echemfem/tree/main/examples>`_.

* **Planar flow reactor** with electroneutral Nernst-Planck. A Butler-Volmer expression is used for the redox reaction, and there are no homogeneous bulk reactions. A custom mesh is used with refinements close to the electrodes.
    
    * `2D case for binary electrolyte <https://github.com/LLNL/echemfem/tree/main/examples/bortels_twoion.py>`_ (:math:`\ce{CuSO4}`)
    * `Nondimensional version of binary case <https://github.com/LLNL/echemfem/tree/main/examples/bortels_twoion_nondim.py>`_ 
    * `2D case for three-ion electrolyte <https://github.com/LLNL/echemfem/tree/main/examples/bortels_threeion.py>`_ (:math:`\ce{CuSO4}` and :math:`\ce{H2SO4}`)
    * `Nondimensional version of three ion case <https://github.com/LLNL/echemfem/tree/main/examples/bortels_threeion_nondim.py>`_ 
    * `3D version of the three-ion case using custom preconditioners for a Discontinuous Galerkin (DG) scheme, and nondimensional <https://github.com/LLNL/echemfem/tree/main/examples/bortels_threeion_extruded_3D_nondim.py>`_ 
    * `A transient version of the 2D case for three-ion electrolyte <https://github.com/LLNL/echemfem/tree/main/examples/bortels_threeion_transient.py>`_

* Simple 1D reaction-diffusion system for :math:`\ce{CO2}` electrolysis in :math:`\ce{KHCO3}`
    
    * `Simplified bicarbonate bulk reactions <https://github.com/LLNL/echemfem/tree/main/examples/gupta.py>`_
    * `Full bicarbonate bulk reactions <https://github.com/LLNL/echemfem/tree/main/examples/carbonate.py>`_
    * `Different implementation of the full bicarbonate bulk reactions <https://github.com/LLNL/echemfem/tree/main/examples/carbonate_homog_params.py>`_ (using the :doc:`homog_params <user_guide/homog_params>` interface)

* 2D flow-past the electrode with advection-diffusion

    * `Two species toy model with shear flow <https://github.com/LLNL/echemfem/tree/main/examples/tworxn.py>`_
    * `CO2 electrolysis in bicarbonate with linear charge-transfer kinetics and shear flow <https://github.com/LLNL/echemfem/tree/main/examples/bicarb.py>`_
    * `Two species toy model with irregular electrode with Navier-Stokes for the flow <https://github.com/LLNL/echemfem/tree/main/examples/tworxn_irregular.py>`_

* 1D model for the :math:`\ce{CO2}` electrolysis in a copper catalyst layer of a gas diffusion electrode (GDE). The model uses electroneutral Nernst-Planck in a porous medium.

    * `Using substitution of the electroneutrality equation to eliminate a species <https://github.com/LLNL/echemfem/tree/main/examples/catalyst_layer_Cu.py>`_
    * `Solving the electroneutrality equation explicitly <https://github.com/LLNL/echemfem/tree/main/examples/catalyst_layer_Cu_full.py>`_

* `A transient 1D example for :math:`\ce{CO2}` electrolysis on copper with a pulsating applied voltage using advection-diffusion-migration <https://github.com/LLNL/echemfem/tree/main/examples/bicarb_Cu_transient_example.py>`_

* `A simple Vanadium flow battery using advection-diffusion-reaction, Poisson for the ionic potential with a predefinied conductivity and Navier-Stokes-Brinkman for the flow <https://github.com/LLNL/echemfem/tree/main/examples/simple_flow_battery.py>`_

* `A symmetric cylindrical pore model for CO2 electrolysis using electroneutral Nernst-Planck and simplified bicarbonate bulk reactions <https://github.com/LLNL/echemfem/tree/main/examples/cylindrical_pore.py>`_

* `A tandem flow-past the electrode system with an Ag catalyst placed in front of a Cu catalyst with electroneutral Nernst-Planck and Tafel kinetics <https://github.com/LLNL/echemfem/tree/main/examples/bicarb_Ag_Cu_tandem_example.py>`_

* `Simple example for the BMCSL model for finite size ion effects <https://github.com/LLNL/echemfem/tree/main/examples/BMCSL.py>`_

* `A 1D example of :math:`\ce{CO2}` electrolysis on copper using the GMPNP model and interpolating artificial data using splines <https://github.com/LLNL/echemfem/tree/main/examples/advanced_examples/CO2R_splines/Cu_1D_CO2R_COR_artificial_splines.py>`_
