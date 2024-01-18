# Copyright (c) 2024 Jeremiah Lübke <jeremiah.luebke@rub.de>,
# Frederic Effenberger, Mike Wilbert, Horst Fichtner, Rainer Grauer
#
# Distributed under the MIT License

import numpy as np
from typing import Union


class VectorUtils:
    def mag(self, in_: str = "res", out: str = "f") -> np.ndarray:
        self._eval(f"sum({in_}**2, 0)", out=out)
        self._eval(f"sqrt({out})", out=out)
        return self._f

    def _normalize_std(self, in_: str = "res") -> np.ndarray:
        self.res /= np.sqrt(
            self._eval(f"sum({in_}**2)")
            / self.grid_size**self.dimension
            / self.components
        )
        return self.res

    def _bwd_vector_potential(self) -> np.ndarray:
        for i in range(self.components):
            self._eval(f"v{i}", out="g")
            self._bwd()
            self._eval("f", out=f"res{i}")
        return self.res

    def _cross(self, a: str, b: str, out: str) -> np.ndarray:
        assert isinstance(a, str) and a in self._variables
        assert isinstance(b, str) and b in self._variables
        assert isinstance(out, str) and out in self._variables
        tmp0 = self._f
        tmp1 = self._g.view(self.ftype.name)
        tmp1 = tmp1.reshape(-1)[: -2 * self.grid_size * self.grid_size]
        tmp1 = tmp1.reshape(self._fwd_tuple)
        assert self._g.__array_interface__["data"] == tmp1.__array_interface__["data"]
        self._eval(f"{a}1 * {b}2 - {a}2 * {b}1", out=tmp0)
        self._eval(f"{a}2 * {b}0 - {a}0 * {b}2", out=tmp1)
        self._eval(f"{a}0 * {b}1 - {a}1 * {b}0", out=f"{out}2")
        self._eval("tmp0", {"tmp0": tmp0}, out=f"{out}0")
        self._eval("tmp1", {"tmp1": tmp1}, out=f"{out}1")
        return self._variables[out]
