// Copyright (c) 2024 Jeremiah Lübke <jeremiah.luebke@rub.de>,
// Frederic Effenberger, Mike Wilbert, Horst Fichtner, Rainer Grauer
//
// Distributed under the MIT License

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <omp.h>

#ifndef real
#define real float
#endif

struct grid_coord
{
    size_t c;
    real d;
};

real dist3sq(const grid_coord &xg, const grid_coord &yg, const grid_coord &zg)
{
    return xg.d * xg.d + yg.d * yg.d + zg.d * zg.d;
}

size_t c2i(const grid_coord &xg, const grid_coord &yg, const grid_coord &zg, size_t x_max)
{
    return xg.c * x_max * x_max + yg.c * x_max + zg.c;
}

grid_coord trunc_coord(real c, real dc, size_t c_max)
{
    c = std::fmod(c / dc, c_max);
    if (c < 0)
        c += c_max;
    size_t c0 = std::trunc(c);
    real d = c - c0;
    return {c0, d};
}

std::vector<grid_coord>
coords_on_grid(real c, real dc, size_t dq, size_t c_max)
{
    std::vector<grid_coord> res;
    grid_coord g = trunc_coord(c, dc, c_max);
    for (size_t q = dq - 1; q > 0; --q)
        res.push_back({g.c < q ? c_max - (q - g.c) : g.c - q, g.d + q});
    res.push_back(g);
    for (size_t q = 1; q <= dq; ++q)
        res.push_back({g.c + q >= c_max ? (g.c + q) - c_max : g.c + q, -g.d + q});
    return res;
}