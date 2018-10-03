# Torontonian Sampling

This repository contains the source code used to produce the results presented in *"Classical benchmarking of Gaussian Boson Sampling on the Titan supercomputer"* [arXiv:1810.00900](https://arxiv.org/abs/1810.00900).

## Contents

This repository contains:

* Fortran source code in the directory `src` which calculates samples from the Torontonian, given a Gaussian state vector of means and covariance matrix.
* This Fortran source code can also be interfaced with Python
* Two examples on how to use the Torontonian sampling module. A Fortran example in the `examples` folder, and an interactive Python Jupyter notebook `TorontonianSampling.ipynb`.

## Installation

The Torontonian sampling Fortran module can be used either via Fortran, or via the Python interface.

### Interfacing via Fortran

If using the module via Fortran, no external dependencies are required, just a Fortran compiler like `gfortran`. On Ubuntu based systems, this can be installed via apt:
```bash
sudo apt install gfortran
```
Then, simply run
```bash
make fortran
```
in the top level directory. The Fortran modules will be compiled, and the modules stored in the directory `include`. To use the module with your own Fortran, simply include the `use torontonian_samples` at the top of the program, and compile the commands
```bash
gfortran -o program program.f90  /path/to/include/*.o -I/path/to/include/
```

See the file `examples/fortran_example.f90` for an example program that uses the Torontonian sampling module. This can be compiled by running `make example` from the top level directory.

### Interfacing via Python

To compile the module for use with Python, `NumPy` is required to be installed, as well as a Fortran compiler such as `gfortran`.
`NumPy` can be installed via `pip`:
```bash
pip install numpy
```
Then, simply run
```bash
make python
```
in the top level directory to compile the Python module. The module `torontonian_samples.cpython-*-.so` will be created, which can then be imported in Python via `import torontonian_samples`.

## Authors

Brajesh Gupt.

If you are doing any research using this source code, please cite the following paper:

> Brajesh Gupt, Juan Miguel Arrazola, Nicolas Quesada, and Thomas R. Bromley.  Classical benchmarking of Gaussian Boson Sampling on the Titan supercomputer, [arXiv:1810.00900](https://arxiv.org/abs/1810.00900)

## License

This source code is free and open source, released under the Apache License, Version 2.0.
