# Synthetic Magnetic Turbulence &mdash; Continuous Cascade & Lagrangian Map

## ACKNOWLEDGEMENT

This is the implementation of the algorithm described in
> J. Lübke, F. Effenberger, M. Wilbert, H. Fichtner and R. Grauer, *Towards Synthetic Magnetic Turbulence with Coherent Structures* (2024).

If the software contributes to findings you decide to present or publish, please be so kind and cite this reference. Thank you!

## LICENSE

This software is distributed under the [MIT License](https://opensource.org/license/mit/).

## INSTALLATION and USAGE

    conda create --name myenv python=3.11
    conda activate myenv
    python -m pip install -r requirements.txt
    python -m pip install numexpr_erf/
    python -c "import numexpr_erf as ne; print(ne.__version__)"

The packages `interp1d`, `interp3d` and `rvs_omp` compile c++ shared libraries on the fly, which requires a recent compiler with OpenMP support.
The compiler and flags are specified in the file `./compiler`.
The code was tested on Linux machines.

After successfull installation of the dependencies, the jupyter notebooks in the `examples/` directory provide basic usage examples.

The heavy use of `numexpr` aims to reduce memory overhead of temporary arrays as much as possible.
This comes sometimes at the cost of reduced readability of the code.

## DEPENDENCIES

This project relies heavily on a modified version of [numexpr](https://github.com/pydata/numexpr), which includes the error function `erf`.
This is achieved by including a copy of the `numexpr-2.8.8` source code with this repository, renaming it to `numexpr_erf`, and adding the following lines:

    numexpr_erf/numexpr_erf/expressions.py:l18 + import scipy.special
    numexpr_erf/numexpr_erf/expressions.py:l375 + 'erf': func(scipy.special.erf, 'float', 'double'),
    numexpr_erf/numexpr_erf/functions.hpp:l39 + FUNC_FF(FUNC_ERF_FF,     "erf_ff",      erff,   erff2,   vsEr)
    numexpr_erf/numexpr_erf/functions.hpp:l84 + FUNC_DD(FUNC_ERF_DD,     "erf_dd",      erf,   vdErf)
    numexpr_erf/numexpr_erf/necompiler.py:l72 + ,"erf"

All further dependencies are listed in `requirements.txt`.

## USER SUPPORT

Contributions, suggestions, and notes on issues are always welcome. Unfortunately, we do not have the ressources to offer individual user support.
