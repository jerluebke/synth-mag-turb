// Copyright (c) 2024 Jeremiah Lübke <jeremiah.luebke@rub.de>,
// Frederic Effenberger, Mike Wilbert, Horst Fichtner, Rainer Grauer
//
// Distributed under the MIT License

#include <algorithm>
#include <cmath>
#include <omp.h>

template <typename Float>
static Float lerp(Float a, Float b, Float t)
{
    if ((a <= 0 && b >= 0) || (a >= 0 && b <= 0))
        return std::fma(t, b, (1.0 - t) * a);
    else
        return std::fma(t, b - a, a);
}

template <typename Float>
static void interp1d(const Float *xp, const Float *yp, Float *x, size_t xp_size, size_t x_size)
{
#pragma omp parallel for
    for (size_t i = 0; i < x_size; ++i)
    {
        const Float *xi = std::lower_bound(xp, xp + xp_size, x[i]);
        size_t idx = std::distance(xp, xi);
        if (idx == 0 || idx == xp_size)
            x[i] = static_cast<Float>(0.0);
        else
        {
            Float d0 = xp[idx] - xp[idx - 1];
            Float d1 = x[i] - xp[idx - 1];
            x[i] = lerp(yp[idx - 1], yp[idx], d1 / d0);
        }
    }
}

extern "C"
{
    void interp1d_double(const double *xp, const double *yp, double *x, size_t xp_size, size_t x_size)
    {
        interp1d(xp, yp, x, xp_size, x_size);
    }

    void interp1d_float(const float *xp, const float *yp, float *x, size_t xp_size, size_t x_size)
    {
        interp1d(xp, yp, x, xp_size, x_size);
    }
}
