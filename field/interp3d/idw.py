# Copyright (c) 2024 Jeremiah Lübke <jeremiah.luebke@rub.de>,
# Frederic Effenberger, Mike Wilbert, Horst Fichtner, Rainer Grauer
#
# Distributed under the MIT License

import os
import sys
import ctypes
import time
import numpy as np
from pathlib import Path


def _get_cfunc(ftype_name):
    from ..utils._get_compiler import compile_cmd

    cftype, npftype, postfix, ftype_cname = {
        "float64": (ctypes.c_double, np.float64, "", "double"),
        "float32": (ctypes.c_float, np.float32, "f", "float"),
    }[ftype_name]

    # compiler = "c++"
    # cflags = "-std=c++17 -Wall -Wextra -O3 -march=native -fPIC -shared -lm -fopenmp"
    path = Path(__file__).parent.resolve()
    lpath = Path(path, f"libidw{postfix}.so").resolve()
    compile_cmd = f"{compile_cmd} -Dreal={ftype_cname} {Path(path, 'idw.cpp').resolve()} -o {lpath}"
    if not lpath.exists():
        print(f"[INFO] Compiling libidw{postfix}.so", file=sys.stderr)
        print(f"[INFO] Running {compile_cmd}", file=sys.stderr)
        os.system(f'/bin/bash -c "{compile_cmd}"')

    lib = ctypes.cdll.LoadLibrary(lpath)
    _f = lib.fwd
    _f.argtype = [
        ctypes.POINTER(cftype),
        ctypes.POINTER(cftype),
        ctypes.POINTER(cftype),
        ctypes.POINTER(cftype),
        ctypes.POINTER(cftype),
        ctypes.POINTER(cftype),
        ctypes.POINTER(cftype),
        ctypes.POINTER(cftype),
        ctypes.POINTER(cftype),
        ctypes.POINTER(cftype),
        cftype,
        cftype,
        ctypes.c_size_t,
    ]

    def _idw(coords, values, grid_spacing, out, *, query_spacing, weights):
        assert coords.dtype == np.dtype(npftype)
        assert values.dtype == np.dtype(npftype)
        assert out.dtype == np.dtype(npftype)
        assert weights.dtype == np.dtype(npftype)
        out[:] = 0
        weights[:] = 0
        xc_ptr = coords[0].ctypes.data_as(ctypes.POINTER(cftype))
        yc_ptr = coords[1].ctypes.data_as(ctypes.POINTER(cftype))
        zc_ptr = coords[2].ctypes.data_as(ctypes.POINTER(cftype))
        valx_ptr = values[2].ctypes.data_as(ctypes.POINTER(cftype))
        valy_ptr = values[1].ctypes.data_as(ctypes.POINTER(cftype))
        valz_ptr = values[0].ctypes.data_as(ctypes.POINTER(cftype))
        outx_ptr = out[2].ctypes.data_as(ctypes.POINTER(cftype))
        outy_ptr = out[1].ctypes.data_as(ctypes.POINTER(cftype))
        outz_ptr = out[0].ctypes.data_as(ctypes.POINTER(cftype))
        weight_ptr = weights.ctypes.data_as(ctypes.POINTER(cftype))
        dx = cftype(grid_spacing)
        dq = cftype(query_spacing)
        x_max = ctypes.c_size_t(coords.shape[1])
        _f(
            xc_ptr,
            yc_ptr,
            zc_ptr,
            valx_ptr,
            valy_ptr,
            valz_ptr,
            outx_ptr,
            outy_ptr,
            outz_ptr,
            weight_ptr,
            dx,
            dq,
            x_max,
        )
        time.sleep(1e-6)
        return out

    return _idw


_func_dict = {key: _get_cfunc(key) for key in ("float32", "float64")}


def idw(*args, **kwds):
    return _func_dict[args[0].dtype.name](*args, **kwds)
