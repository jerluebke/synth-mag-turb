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

# compiler = "c++"
# cflags = "-Wall -Wextra -O3 -march=native -fPIC -shared -lm -fopenmp"
path = Path(__file__).parent.resolve()
compile_cmd = f"{compile_cmd} {Path(path, 'rvs_omp.cpp').resolve()} -o {Path(path, 'librvs_omp.so')}"
if not Path(path, "librvs_omp.so").exists():
    print("[INFO] Compiling librvs_omp.so", file=sys.stderr)
    print(f"[INFO] Running {compile_cmd}", file=sys.stderr)
    os.system(f'/bin/bash -c "{compile_cmd}"')

libutils = ctypes.cdll.LoadLibrary(os.path.join(path, "librvs_omp.so"))
_normal_rvs_dbl = libutils.normal_rvs_double
_normal_rvs_dbl.argtypes = [
    ctypes.c_uint,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_double,
    ctypes.c_double,
]
_normal_rvs_flt = libutils.normal_rvs_float
_normal_rvs_flt.argtypes = [
    ctypes.c_uint,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_float,
    ctypes.c_float,
]
_uniform_rvs_dbl = libutils.uniform_rvs_double
_uniform_rvs_dbl.argtypes = [
    ctypes.c_uint,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_double,
    ctypes.c_double,
]
_uniform_rvs_flt = libutils.uniform_rvs_float
_uniform_rvs_flt.argtypes = [
    ctypes.c_uint,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_float,
    ctypes.c_float,
]


def rvs(gen_dbl, gen_flt):
    def _rvs(out, p1=0.0, p2=1.0, seed=None):
        if seed is None:
            seed = np.random.randint(np.iinfo("uint32").max)
        gen, ftype = {
            "float64": (gen_dbl, ctypes.c_double),
            "float32": (gen_flt, ctypes.c_float),
        }[out.dtype.name]
        seed = ctypes.c_uint(seed)
        size = ctypes.c_size_t(out.size)
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ftype))
        p1 = ftype(p1)
        p2 = ftype(p2)
        gen(seed, out_ptr, size, p1, p2)
        return out

    return _rvs


normal_rvs = rvs(_normal_rvs_dbl, _normal_rvs_flt)
uniform_rvs = rvs(_uniform_rvs_dbl, _uniform_rvs_flt)
