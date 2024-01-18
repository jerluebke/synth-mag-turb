# Copyright (c) 2024 Jeremiah Lübke <jeremiah.luebke@rub.de>,
# Frederic Effenberger, Mike Wilbert, Horst Fichtner, Rainer Grauer
#
# Distributed under the MIT License

import numpy as np
from typing import Union


class Statistics:
    def kbins(
        self,
        lo: float = None,
        hi: float = None,
        num: int = None,
        prepend_zero: bool = False,
    ) -> np.ndarray:
        lo = lo or 1.0
        hi = hi or self.grid_size // 2
        num = num or self.grid_size // 4
        kbins = np.unique(np.geomspace(lo, hi, num).astype(int))
        if prepend_zero:
            kbins = np.insert(kbins, 0, 0)
        return kbins, (kbins[1:] + kbins[:-1]) / 2

    def spectrum(
        self, bins: Union[np.ndarray, int], kmag: Union[np.ndarray, str] = None
    ) -> list:
        print("computing radial spectra", end="")
        S_list = []
        kmag = self._eval(f"sqrt{self._kmag_squared}", out=kmag)
        for i in range(self.components):
            self._eval(f"res{i}", out="f")
            self._fwd()
            self._eval("abs(g)**2", out=self._g)
            S_list += [self._radial(bins, kmag)]
        print(".")
        return S_list

    def _radial(self, bins: Union[np.ndarray, int], kmag: np.ndarray) -> np.ndarray:
        S = np.histogram(kmag, bins=bins, weights=self._g.real, density=True)[0]
        return S
