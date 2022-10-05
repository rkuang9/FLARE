//
// Created by RKuang on 9/24/2022.
//

#ifndef ORION_GLOROT_HPP
#define ORION_GLOROT_HPP

#include "weights.hpp"

namespace orion
{

template<Eigen::Index Rank = 2>
class GlorotUniform : public Initializer<Rank>
{
public:
    /**
     * Glorot uniform initialization, creates a tensor on a uniform distribution
     * in range +- sqrt(6 / (fan_in + fan_out))
     *
     * @param dims      tensor dimensions e.g. Tensor<2>::Dimensions(2, 4)
     * @param fan_in    number of layer input units
     * @param fan_out   number of layer output units
     * @return Tensor<Rank>
     */
    Tensor<Rank> Initialize(typename Tensor<Rank>::Dimensions dims,
                         int fan_in, int fan_out) const override
    {
        Scalar limit = std::sqrt(6.0 / (fan_in + fan_out));
        return RandomUniform<2>(dims, -limit, limit);
    }


    /**
     * Rank 3 equivalent
     */
    /*Tensor<3> Initialize(Tensor<3>::Dimensions dims,
                         int fan_in, int fan_out) const override
    {
        Scalar limit = std::sqrt(6.0 / (fan_in + fan_out));
        return RandomUniform<3>(dims, -limit, limit);
    }*/
};


template<Eigen::Index Rank = 2>
class GlorotNormal : public Initializer<Rank>
{
public:
    /**
     * Glorot normal initialization, creates a tensor on a real distribution
     * with mean 0 and standard deviation sqrt(2 / (fan_in + fan_out))
     *
     * @param dims      tensor dimensions e.g. Tensor<2>::Dimensions(2, 4)
     * @param fan_in    number of layer input units
     * @param fan_out   number of layer output units
     * @return Tensor<3>
     */
    Tensor<Rank> Initialize(typename Tensor<Rank>::Dimensions dims,
                         int fan_in, int fan_out) const override
    {
        Scalar stddev = std::sqrt(2.0 / (fan_in + fan_out));
        return RandomNormal<2>(dims, 0.0, stddev);
    }


    /**
     * Rank 3 equivalent
     */
    /*Tensor<3> Initialize(Tensor<3>::Dimensions dims,
                         int fan_in, int fan_out) const override
    {
        Scalar stddev = std::sqrt(2.0 / (fan_in + fan_out));
        return RandomNormal<3>(dims, 0.0, stddev);
    }*/
};

}

#endif //ORION_GLOROT_HPP
