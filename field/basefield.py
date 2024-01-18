# Copyright (c) 2024 Jeremiah Lübke <jeremiah.luebke@rub.de>,
# Frederic Effenberger, Mike Wilbert, Horst Fichtner, Rainer Grauer
#
# Distributed under the MIT License

print(
    """Copyright (c) 2024 Jeremiah Lübke <jeremiah.luebke@rub.de>,
Frederic Effenberger, Mike Wilbert, Horst Fichtner, Rainer Grauer

Distributed under the MIT License.
 
This is the implementation of the algorithm described in
 > J. Lübke, F. Effenberger, M. Wilbert, H. Fichtner and R. Grauer,
 > Towards Synthetic Magnetic Turbulence with Coherent Structures
 > (2024).

If the software contributes to findings you decide to present or publish,
please be so kind and cite this reference. Thank you!
"""
)

import os
import numpy as np
import numexpr_erf as ne
import pyfftw
import pickle
from enum import Enum
from pathlib import Path
from typing import Union
from .utils.derivatives import Derivatives
from .utils.fieldio import FieldIO, _get_writer_kwds
from .utils.statistics import Statistics
from .utils.vectorutils import VectorUtils


def export_pyfftw_wisdom(filename):
    with open(filename, "w+b") as fp:
        pickle.dump(pyfftw.export_wisdom(), fp)


def import_pyfftw_wisdom(filename):
    with open(filename, "rb") as fp:
        pyfftw.import_wisdom(pickle.load(fp))


class Precision(Enum):
    SINGLE = (np.dtype("float32"), np.dtype("complex64"))
    DOUBLE = (np.dtype("float64"), np.dtype("complex128"))


class BaseField(Derivatives, FieldIO, Statistics, VectorUtils):
    _kind = None

    def __init__(
        self,
        name: str,
        grid_size: int,
        *,
        dimension: int,
        components: int,
        L_box: float = 1.0,
        precision: Precision = Precision.DOUBLE,
        num_threads: int = None,
        wisdom_path: str = None,
        init_pyfftw: bool = True,
    ):
        self.name = name
        self.wisdom_path = wisdom_path
        self.num_threads = num_threads or os.cpu_count()
        ne.set_num_threads(self.num_threads)
        self.precision = precision
        self.ftype, self.ctype = precision.value
        self.dimension = dimension
        self.components = components
        self.grid_size = grid_size
        self._dx = 1 / grid_size
        self.dx = L_box * self._dx
        self.L_box = L_box
        kx = np.fft.fftfreq(grid_size, self._dx).astype(self.ftype)
        ki_list = [kx] * (dimension - 1) + [kx[: grid_size // 2 + 1]]
        self._ki = np.meshgrid(*ki_list, sparse=True, copy=False, indexing="ij")
        self._origin = tuple([0] * dimension)
        self._fwd_tuple = tuple([grid_size] * dimension)
        self._bwd_tuple = tuple([grid_size] * (dimension - 1) + [grid_size // 2 + 1])
        self._vfwd_tuple = tuple([components] + [grid_size] * dimension)
        self._vbwd_tuple = tuple(
            [components] + [grid_size] * (dimension - 1) + [grid_size // 2 + 1]
        )
        self.res = np.zeros(self._vfwd_tuple, dtype=self.ftype)
        self._variables = (
            {
                "dim": dimension,
                "n": grid_size,
                "res": self.res,
            }
            | {f"k{c}": self._ki[i] for i, c in enumerate("xyz"[:dimension])}
            | {f"res{i}": self.res[i] for i in range(self.components)}
        )

        if init_pyfftw:
            self._f = pyfftw.empty_aligned(self._fwd_tuple, dtype=self.ftype)
            self._g = pyfftw.empty_aligned(self._bwd_tuple, dtype=self.ctype)
            self._variables |= {"f": self._f, "g": self._g}

            wisdom_file = Path(
                wisdom_path or "wisdom",
                f"wisdom-{dimension}D-{grid_size}n-{self.num_threads}threads-{self.ctype.name}",
            )
            if wisdom_file.exists():
                import_pyfftw_wisdom(wisdom_file)
                flags = ("FFTW_WISDOM_ONLY",)
                print(
                    "initializing FFTW with wisdom from disk (if this step fails: delete wisdom file)",
                    end="",
                )
            else:
                wisdom_file.parent.mkdir(parents=True, exist_ok=True)
                pyfftw.forget_wisdom()
                flags = ("FFTW_MEASURE",)
                print("initializing FFTW, generating wisdom", end="")
            self._fwd = pyfftw.FFTW(
                self._f,
                self._g,
                axes=tuple(range(dimension)),
                direction="FFTW_FORWARD",
                threads=self.num_threads,
                flags=flags,
            )
            self._bwd = pyfftw.FFTW(
                self._g,
                self._f,
                axes=tuple(range(dimension)),
                direction="FFTW_BACKWARD",
                threads=self.num_threads,
                flags=flags,
            )
            if flags[0] == "FFTW_MEASURE":
                print(f". writing new wisdom to {wisdom_file}", end="")
                export_pyfftw_wisdom(wisdom_file)
            print(".")

    def _eval(
        self,
        expr: str,
        extra_variables: dict = None,
        out: Union[np.ndarray, str] = None,
    ) -> np.ndarray:
        variables = self._variables | (
            {} if extra_variables is None else extra_variables
        )
        if isinstance(out, str):
            out = self._variables[out]
        return ne.evaluate(expr, variables, out=out, casting="same_kind")

    def __call__(self, *args, **kwds) -> np.ndarray:
        write_field, writer_kwds = _get_writer_kwds(kwds)
        self._call_impl(*args, **kwds)
        self._normalize_std()
        if write_field:
            self.write_field(*args, **writer_kwds, **kwds)
        return self.res
