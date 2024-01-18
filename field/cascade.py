# Copyright (c) 2024 Jeremiah Lübke <jeremiah.luebke@rub.de>,
# Frederic Effenberger, Mike Wilbert, Horst Fichtner, Rainer Grauer
#
# Distributed under the MIT License

import numpy as np
from functools import partial
from .rvs_omp.rvs_omp import normal_rvs, uniform_rvs
from .basefield import BaseField, Precision
from .interp3d.idw import idw


class Cascade3D(BaseField):
    _kmag_squared = "(kx**2+ky**2+kz**2)"
    _kind = "intermittent"

    def __init__(self, name: str, grid_size: int, **kwds):
        super().__init__(name, grid_size, dimension=3, components=3, **kwds)
        self._cd = np.pi / 6.0
        self._e = np.zeros(self._vfwd_tuple, dtype=self.ftype)
        self._v = np.zeros(self._vbwd_tuple, dtype=self.ctype)
        self._variables |= {
            "e": self._e,
            "v": self._v,
            "omega": self._e[0],
            "theta": self._e[1],
            "phi": self._e[2],
            **{f"e{i}": self._e[i] for i in range(self.components)},
            **{f"v{i}": self._v[i] for i in range(self.components)},
        }

    def _s(self, i: int, N: int, L: float):
        s_N = L / self.L_box
        return self._dx * (s_N / self._dx) ** (i / N)

    def _call_impl(
        self,
        number_of_modes: int,
        correlation_length: float,
        spectral_index: float,
        intermittency_parameter: float,
        *,
        apply_curl: bool = True,
        notify_done: bool = True,
    ) -> np.ndarray:
        print(f"running cascade: {self.name}.")
        self.res[:] = 0
        self._e[:] = 0
        self._v[:] = 0
        for i in range(number_of_modes, 0, -1):
            scale = self._s(i, number_of_modes, correlation_length)
            ds = scale - self._s(i - 1, number_of_modes, correlation_length)
            variance = (
                self._cd * intermittency_parameter * ds / scale ** (self.dimension + 1)
            )
            scalefactor = ds * scale ** (spectral_index - self.dimension)
            self._generate_step(scale, variance, scalefactor)
        if apply_curl:
            print("transforming to real space and applying curl.")
            self._curl()
        if notify_done:
            print("done.")
        return self.res

    def _generate_step(self, scale, variance, scalefactor, *, end="\n"):
        print(f"running scale {scale:g}. generating omega", end="")
        self._gaussian_noise("omega", scale, -variance / 2, variance, accumulate=True)
        print(", theta", end="")
        self._gaussian_noise("theta", scale, 0, 1)
        print(", phi", end="")
        self._gaussian_noise("phi", scale, 0, 1)
        print(". normalizing", end="")
        self._normalize_noise()
        print(". wavelet step", end="")
        self._wavelet_convolution(scale, scalefactor)
        print(".", end=end)

    def _gaussian_noise(self, name, scale, mean, variance, accumulate=False):
        indicator = f"(scale*n)**dim*exp(-{self._kmag_squared}*scale**2)"
        normal_rvs(self._g.view(self.ftype), 0, np.sqrt(variance))
        self._g[self._origin] = mean
        self._eval(f"g*{indicator}", {"scale": scale}, out="g")
        self._bwd()
        self._eval(f"{name}+f" if accumulate else "f", out=name)

    def _normalize_noise(self):
        funcs = {
            "theta": "arccos(-erf({}/sqrt(2)))",
            "phi": f"{np.pi}*(1+erf({{}}/sqrt(2)))",
        }
        for name, func in funcs.items():
            std = np.sqrt(
                self._eval(f"sum({name}**2)") / self.grid_size**self.dimension
            )
            func = func.format(f"{name}/{std}")
            self._eval(func, out=name)

    def _wavelet_convolution(self, scale, scalefactor):
        wavelet = (
            f"scale**(dim+2)*{self._kmag_squared}*exp(-{self._kmag_squared}*scale**2)"
        )
        rij = [
            "sin(theta)*cos(phi)",
            "sin(theta)*sin(phi)",
            "cos(theta)",
        ]
        for k in range(3):
            self._eval(f"exp(omega)*({rij[k]})", out="f")
            self._fwd()
            self._eval(
                f"v{k}+scalefactor*g*{wavelet}",
                {"scale": scale, "scalefactor": scalefactor},
                out=f"v{k}",
            )

    def randomize_phases(self, **kwds) -> np.ndarray:
        from .basefield import _get_writer_kwds

        print("randomizing phases", end="")
        self.res[:] = 0
        for i in range(self.components):
            uniform_rvs(self._g.view(self.ftype), -np.pi, np.pi)
            self._eval(f"abs(v{i})*exp(1j*real(g))", out="g")
            self._bwd()
            self._curl_step(i)
        self._normalize_std()
        print(".")
        write_field, writer_kwds = _get_writer_kwds(kwds)
        if write_field:
            self._kind = "randomphases"
            self.write_field(**writer_kwds)
            self._kind = type(self)._kind
        return self.res


class LagrangianMapping3D(Cascade3D):
    _kind = "lagrangian_mapping"

    def __init__(
        self, name: str, grid_size: int, cfl: float, query_spacing: float = 2, **kwds
    ):
        super().__init__(name, grid_size, **kwds)
        assert cfl != 0
        self.cfl = cfl
        self._interp3d = partial(idw, query_spacing=query_spacing, weights=self._f)
        self._c = np.zeros(self._vfwd_tuple, dtype=self.ftype)
        self._reset_coords()
        self._variables |= {"c": self._c}

    def _reset_coords(self):
        x = np.arange(0, self.grid_size) * self.dx
        self._c[0] = x[:, np.newaxis, np.newaxis]
        self._c[1] = x[np.newaxis, :, np.newaxis]
        self._c[2] = x[np.newaxis, np.newaxis, :]
        self._fix_coords = False

    def _call_impl(
        self, *args, other: bool = False, lowpass_kwds: bool = None, **kwds
    ) -> np.ndarray:
        if new_cfl := kwds.pop("cfl", False):
            self.cfl = new_cfl
        self._reset_coords()
        super()._call_impl(*args, apply_curl=False, notify_done=False, **kwds)
        if other:
            self._fix_coords = True
            super()._call_impl(*args, apply_curl=False, notify_done=False, **kwds)
        self._transform_and_interpolate_grid(lowpass_kwds or {})
        print("done.")
        return self.res

    def _generate_step(self, scale, variance, scalefactor):
        if not self._fix_coords:
            super()._generate_step(scale, variance, scalefactor, end=" ")
            print("advecting coordinates", end="")
            self._curl()
            self.mag()
            norm = np.max(self._f)
            self._eval(f"c+{self.cfl:g}*{scale:g}*res/{norm:g}", out="c")
            print(".")
        else:
            super()._generate_step(scale, variance, scalefactor, end="\n")

    def _transform_and_interpolate_grid(self, lowpass_kwds):
        print("transforming to real space", end="")
        self._bwd_vector_potential()
        for i in range(self.components):
            self._c[i].sort(axis=i)
        print(".")
        self._interp3d(self._c, self.res, self.dx, self._e)
        print("applying curl", end="")
        self.res[:] = 0
        for i in range(self.components):
            self._eval(f"e{i}", out="f")
            self._low_pass(i, **lowpass_kwds)
            self._curl_step(i)
        print(".")

    def _low_pass(self, i, *, k0=None, k1=None, p0=0):
        k0 = k0 or self.grid_size // 2
        k1 = k1 or self.grid_size // 2
        print(f". lowpass filtering with {k0=}, {k1=}, {p0=}", end="")
        self._fwd()
        self._eval(
            f"g*{self._kmag_squared}**(p0/2.0)*exp(-{self._kmag_squared}/k0**2/2.0)",
            {"k0": k0, "p0": p0},
            out="g",
        )
        self._eval(f"where({self._kmag_squared}>k1**2, 0.0, g)", {"k1": k1}, out="g")
        self._g[0, 0, 0] = 0.0
        self._eval("g", out=f"v{i}")
        self._bwd()
