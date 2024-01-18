// Copyright (c) 2024 Jeremiah Lübke <jeremiah.luebke@rub.de>,
// Frederic Effenberger, Mike Wilbert, Horst Fichtner, Rainer Grauer
//
// Distributed under the MIT License

#include "common.hpp"

extern "C"
{
    void fwd(
        const real *xc,
        const real *yc,
        const real *zc,
        const real *ax,
        const real *ay,
        const real *az,
        real *resx,
        real *resy,
        real *resz,
        real *weights,
        real dx,
        real dq,
        size_t x_max)
    {
        std::cout << "running forward interpolation" << std::flush;
        real eps = std::numeric_limits<real>::epsilon();
        size_t size = x_max * x_max * x_max;
#pragma omp parallel
        {
#pragma omp for
            for (size_t id = 0; id < size; ++id)
            {
                auto xgrid = coords_on_grid(xc[id], dx, std::ceil(std::abs(dq)), x_max);
                auto ygrid = coords_on_grid(yc[id], dx, std::ceil(std::abs(dq)), x_max);
                auto zgrid = coords_on_grid(zc[id], dx, std::ceil(std::abs(dq)), x_max);
                for (const auto &xg : xgrid)
                    for (const auto &yg : ygrid)
                        for (const auto &zg : zgrid)
                            if (real dsq = dist3sq(xg, yg, zg); dsq < dq * dq)
                            {
                                size_t idc = c2i(xg, yg, zg, x_max);
                                real weight = real{1.0} / std::sqrt(dsq + eps * eps);
                                resx[idc] += weight * ax[id];
                                resy[idc] += weight * ay[id];
                                resz[idc] += weight * az[id];
                                weights[idc] += weight;
                            }
            }
#pragma omp barrier
#pragma omp for
            for (size_t id = 0; id < size; ++id)
            {
                if (weights[id] != 0)
                {
                    resx[id] /= weights[id];
                    resy[id] /= weights[id];
                    resz[id] /= weights[id];
                }
            }
        }
        std::cout << "." << std::endl;
    }
}
