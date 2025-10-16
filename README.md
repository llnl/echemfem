# EchemFEM

[![DOI](https://zenodo.org/badge/513600791.svg)](https://zenodo.org/badge/latestdoi/513600791)

[![DOI](https://joss.theoj.org/papers/10.21105/joss.06531/status.svg)](https://doi.org/10.21105/joss.06531)

This code provides Finite Element solvers for electrochemical transport.
Both continuous Galerkin (CG) and discontinuous Galerkin (DG) schemes are provided. The DG scheme for electroneutral Nernst-Planck is described in [Roy et al., 2023](https://doi.org/10.1016/j.jcp.2022.111859). The CG scheme uses SUPG for the advection-migration term.

The following transport mechanisms are available: diffusion, advection, electromigration. EchemFEM supports both non-porous and porous cases. The ionic potential can either be described using an electroneutrality constraint or a Poisson equation. In the porous case, the electronic potential can be described by a Poisson equation.
Some finite size effects are also implemented. For example, the generalized modified Poisson-Nernst-Planck model (GMPNP) is used in the examples of [FireCat](https://github.com/LLNL/firecat), coupling it with a microkinetics model.


LLNL-CODE-837342

## Documentation

User guide and API documentation can be found [here](https://software.llnl.gov/echemfem/index.html).

## Installation

Please install the open-source finite element library [Firedrake](https://www.firedrakeproject.org/download.html).

To install EchemFEM, simply run the following in the parent echemfem folder:
```
pip install -e .
```
The documentation has more details about installation and running the code.

## Citing

To cite EchemFEM, please cite the [JOSS paper](https://doi.org/10.21105/joss.06531). For the DG discretization, please cite the [JCP paper](https://doi.org/10.1016/j.jcp.2022.111859).
