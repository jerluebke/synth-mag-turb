# Copyright (c) 2024 Jeremiah Lübke <jeremiah.luebke@rub.de>,
# Frederic Effenberger, Mike Wilbert, Horst Fichtner, Rainer Grauer
#
# Distributed under the MIT License


class Derivatives:
    @staticmethod
    def _slice_tuple_func(n):
        def _s(i, lo, hi=None):
            sl = [slice(None)] * n
            sl[i] = lo if hi is None else slice(lo, hi)
            return tuple(sl)

        return _s

    def _fd_inplace(self, arr, i, out="f", sign=1):
        arr = self._variables[arr] if isinstance(arr, str) else arr
        out = self._variables[out] if isinstance(out, str) else out
        end = arr.shape[i]
        s = self._slice_tuple_func(arr.ndim)
        out[s(i, 1, -1)] -= (
            sign * (arr[s(i, 0, -2)] - arr[s(i, 2, end)]) / (2 * self.dx)
        )
        out[s(i, 0)] -= sign * (arr[s(i, -1)] - arr[s(i, 1)]) / (2 * self.dx)
        out[s(i, -1)] -= sign * (arr[s(i, -2)] - arr[s(i, 0)]) / (2 * self.dx)
        return out

    def _fd(self, arr, i, out="f"):
        arr = self._variables[arr] if isinstance(arr, str) else arr
        out = self._variables[out] if isinstance(out, str) else out
        end = arr.shape[i]
        s = self._slice_tuple_func(arr.ndim)
        out[s(i, 1, -1)] = -(arr[s(i, 0, -2)] - arr[s(i, 2, end)]) / (2 * self.dx)
        out[s(i, 0)] = -(arr[s(i, -1)] - arr[s(i, 1)]) / (2 * self.dx)
        out[s(i, -1)] = -(arr[s(i, -2)] - arr[s(i, 0)]) / (2 * self.dx)
        return out

    def _spec_diff(self, arr, i):
        self._f[:] = arr
        self._fwd()
        self._g[:] *= 1j * self._ki[i]
        self._bwd()
        return self._f

    def div(self, out):
        assert isinstance(out, str) and out in self._variables
        print("computing div", end="")
        self._f[:] = 0.0
        for i in range(self.components):
            self._fd_inplace(f"res{i}", i, out="f")
        print(".")
        return self._f

    def _curl_step(self, i, in_="f", out="res"):
        j, k = {0: (2, 1), 1: (0, 2), 2: (1, 0)}[i]
        self._fd_inplace(in_, j, f"{out}{k}", +1)
        self._fd_inplace(in_, k, f"{out}{j}", -1)

    def _curl(self):
        self.res[:] = 0.0
        for i in range(self.components):
            self._eval(f"v{i}", out="g")
            self._bwd()
            self._curl_step(i)
        return self.res

    def curl(self, out):
        assert isinstance(out, str) and out in self._variables
        print("computing curl", end="")
        self._variables[out][:] = 0.0
        for i in range(self.components):
            self._curl_step(i, in_=f"res{i}", out=out)
        print(".")
        return self._variables[out]

    def curv(self, out):
        assert isinstance(out, str) and out in self._variables
        print("computing curvature", end="")
        self._variables[out][:] = 0.0
        for i in range(self.components):
            for j in range(self.components):
                self._fd(f"res{i}", j, out="f")
                self._eval(f"{out}{i} + res{j} * f", out=f"{out}{i}")
        self.mag(out="f")
        for i in range(self.components):
            self._eval(f"{out}{i} / f**3", out=f"{out}{i}")
        self._cross("res", out, out=out)
        self._eval(f"sum({out}**2, 0)", out="f")
        self._eval(f"sqrt(f)", out="f")
        print(".")
        return self._f
