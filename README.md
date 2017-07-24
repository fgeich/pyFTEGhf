# pyFTEGhf
Python script solving the Hartree-Fock equations for the electron gas at finite temperature.

## Synopsis

pyFTEGhf is a simple Python code solving the self-consistency equation for the (spin-unpolarized) uniform electron gas in three dimensions at finite temperatures.

## Code Example

Some sample Python scripts using pyFTEGhf are provided in the run folder.

## Motivation

This code was used to generate the Hartree-Fock results for the heat capacity and entropy of the three-dimensional electron gas published in Phys. Rev. B 96, 035132 (2017).

If you use this code to generate data for publications please cite:

"Effective mass of quasiparticles from thermodynamics"  
F. G. Eich, Markus Holzmann, and G. Vignale  
Phys. Rev. B 96, 035132 (2017)
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.035132

## Installation

Simply clone the git repository. The code relies on scipy/numpy support.

## Documentation

A quick overview over the code and the derivation of the algorithm for the solution of the Hartree-Fock equations is presented in pyFTEGhf.pdf located in the doc folder.  

## Acknowledgements

This code was developed as part of the project "ThermalDFT" funded by the European Union’s Framework Programme for Research and Innovation Horizon 2020 (2014-2020) under the Marie Sklodowska-Curie Grant Agreement No. 701796.

## License

Code released under version 3 of the GNU General Public License.
