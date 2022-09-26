//
// Created by macross on 8/7/22.
//

#ifndef ORION_WEIGHTS_HPP
#define ORION_WEIGHTS_HPP

#include "orion/orion_types.hpp"

namespace orion
{

/**
 * Base class for weight initialization
 */
class Initializer
{
public:
    virtual Tensor<2> Initialize(Tensor<2>::Dimensions dims,
                                 int fan_in, int fan_out) const = 0;

    virtual Tensor<3> Initialize(Tensor<3>::Dimensions dims,
                                 int fan_in, int fan_out) const = 0;
};


/**
 * Creates a tensor on a uniform distribution within range min, max
 * Used by the various weight initializers
 *
 * @tparam Rank   tensor rank
 * @param dims    tensor dimensions
 * @param min     maximum value
 * @param max     minimum value
 * @return Tensor<Rank>
 */
template<int Rank>
Tensor<Rank> RandomUniform(typename Tensor<Rank>::Dimensions dims,
                           Scalar min, Scalar max)
{
    std::random_device random;
    std::mt19937_64 mt(random());

    return Tensor<Rank>(dims).template nullaryExpr([&]() {
        return std::uniform_real_distribution<Scalar>(min, max)(mt);
    });
}


/**
 * Creates a tensor on a real distribution within with mean 0, standard deviation stddev
 * Used by the various weight initializers
 *
 * @tparam Rank    tensor rank
 * @param dims     tensor dimensions
 * @param mean     mean value
 * @param stddev   standard deviation
 * @return
 */
template<int Rank>
Tensor<Rank> RandomNormal(typename Tensor<Rank>::Dimensions dims,
                          Scalar mean, Scalar stddev)
{
    std::random_device random;
    std::mt19937_64 mt(random());

    return Tensor<Rank>(dims).template nullaryExpr([&]() {
        return std::normal_distribution<Scalar>(mean, stddev)(mt);
    });
}

} // namespace orion

#endif //ORION_WEIGHTS_HPP
