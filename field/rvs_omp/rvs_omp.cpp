// Copyright (c) 2024 Jeremiah Lübke <jeremiah.luebke@rub.de>,
// Frederic Effenberger, Mike Wilbert, Horst Fichtner, Rainer Grauer
//
// Distributed under the MIT License

#include <random>
#include <cstdint>
#include <cmath>
#include <omp.h>

template<typename Distr, typename Float>
static void fill_array_with_random_numbers(unsigned int seed, Float* res, size_t size, Distr& dist)
{
    std::seed_seq seq{seed};
    std::vector<std::uint32_t> seeds(omp_get_max_threads());
    seq.generate(seeds.begin(), seeds.end());
    std::mt19937 gen;

    #pragma omp parallel firstprivate(gen, dist)
    {
        gen.seed(seeds[omp_get_thread_num()]);
        #pragma omp for
        for ( size_t i = 0; i < size; ++i )
            res[i] = dist(gen);
    }
}

extern "C" {
    void normal_rvs_double(unsigned int seed, double* res, size_t size, double mean, double sigma)
    {
        std::normal_distribution<double> norm(mean, sigma);
        fill_array_with_random_numbers(seed, res, size, norm);
    }
    
    void normal_rvs_float(unsigned int seed, float* res, size_t size, float mean, float sigma)
    {
        std::normal_distribution<float> norm(mean, sigma);
        fill_array_with_random_numbers(seed, res, size, norm);
    }
    
    void uniform_rvs_double(unsigned int seed, double* res, size_t size, double min, double max)
    {
        std::uniform_real_distribution<double> uni(min, max);
        fill_array_with_random_numbers(seed, res, size, uni);
    }
    
    void uniform_rvs_float(unsigned int seed, float* res, size_t size, float min, float max)
    {
        std::uniform_real_distribution<float> uni(min, max);
        fill_array_with_random_numbers(seed, res, size, uni);
    }
}
