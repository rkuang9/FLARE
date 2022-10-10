//
// Created by RKuang on 9/24/2022.
//

#ifndef ORION_LECUN_HPP
#define ORION_LECUN_HPP

#include "weights.hpp"

namespace orion
{

template <int TensorRank>
class LecunUniform : public Initializer<TensorRank>
{
public:
    /**
     * Lecun uniform initialization, creates a tensor on a uniform distribution
     * in range +- sqrt(3 / fan_in)
     *
     * @param dims      tensor dimensions e.g. Tensor<2>::Dimensions(2, 4)
     * @param fan_in    number of layer input units
     * @param fan_out   number of layer output units
     * @return Tensor<2>
     */
    Tensor<TensorRank> Initialize(const Dims<TensorRank> &dims,
                         int fan_in, int fan_out) const override
    {
        Scalar limit = std::sqrt(3.0 / fan_in);
        return RandomUniform<TensorRank>(dims, -limit, limit);
    }
};


template<int TensorRank>
class LecunNormal : public Initializer<TensorRank>
{
public:
    /**
     * Lecun normal initialization, creates a tensor on a real distribution
     * with mean 0 and standard deviation sqrt(1 / fan_in)
     *
     * @param dims      tensor dimensions e.g. Tensor<2>::Dimensions(2, 4)
     * @param fan_in    number of layer input units
     * @param fan_out   number of layer output units
     * @return Tensor<3>
     */
    Tensor<TensorRank> Initialize(const Dims<TensorRank> &dims,
                         int fan_in, int fan_out) const override
    {
        Scalar stddev = std::sqrt(1.0 / fan_in);
        return RandomNormal<TensorRank>(dims, 0.0, stddev);
    }
};

}

#endif //ORION_LECUN_HPP
