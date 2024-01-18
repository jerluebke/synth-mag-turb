# Copyright (c) 2024 Jeremiah Lübke <jeremiah.luebke@rub.de>,
# Frederic Effenberger, Mike Wilbert, Horst Fichtner, Rainer Grauer
#
# Distributed under the MIT License

import os
import sys
import ctypes
import numpy as np
from pathlib import Path
from ..utils._get_compiler import compile_cmd

name = "interp1d"
# compiler = "c++"
# cflags = "-std=c++17 -Wall -Wextra -O3 -march=native -fPIC -shared -lm -fopenmp"
path = Path(__file__).parent.resolve()
compile_cmd = f"{compile_cmd} {Path(path, f'{name}.cpp').resolve()} -o {Path(path, f'lib{name}_omp.so')}"
if not Path(path, f"lib{name}_omp.so").exists():
    print(f"[INFO] Compiling lib{name}_omp.so", file=sys.stderr)
    print(f"[INFO] Running {compile_cmd}", file=sys.stderr)
    os.system(f'/bin/bash -c "{compile_cmd}"')

lib = ctypes.cdll.LoadLibrary(os.path.join(path, f"lib{name}_omp.so"))
_interp1d_dbl = lib.interp1d_double
_interp1d_dbl.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_interp1d_flt = lib.interp1d_float
_interp1d_flt.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
]


def interp1d(xp, yp, xnew):
    func, ftype = {
        "float64": (_interp1d_dbl, ctypes.c_double),
        "float32": (_interp1d_flt, ctypes.c_float),
    }[xnew.dtype.name]
    xp_size = ctypes.c_size_t(xp.size)
    xnew_size = ctypes.c_size_t(xnew.size)
    xp_ptr = xp.ctypes.data_as(ctypes.POINTER(ftype))
    yp_ptr = yp.ctypes.data_as(ctypes.POINTER(ftype))
    xnew_ptr = xnew.ctypes.data_as(ctypes.POINTER(ftype))
    func(xp_ptr, yp_ptr, xnew_ptr, xp_size, xnew_size)
    return xnew
