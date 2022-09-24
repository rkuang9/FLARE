//
// Created by RKuang on 9/22/2022.
//

#ifndef ORION_SWISH_HPP
#define ORION_SWISH_HPP

#include "orion/orion_types.hpp"
#include "sigmoid.hpp"

namespace orion
{
// TODO: prone to exploding gradients, need to clip optimizers
class Swish
{
public:
    static Tensor<2> Activate(const Tensor<2> &features)
    {
        // a = z/(1 + e^-z) = z * sigmoid(z)
        return features * features.sigmoid();
    }


    static Tensor<3> Activate(const Tensor<3> &features)
    {
        return features * features.sigmoid();
    }


    static Tensor<2> Gradients(const Tensor<2> &features)
    {
        Tensor<2> swish = features * features.sigmoid();
        return swish + features.sigmoid() * (1.0 - swish);
    }


    static Tensor<3> Gradients(const Tensor<3> &features)
    {
        Tensor<3> swish = features * features.sigmoid();
        return swish + features.sigmoid() * (1.0 - swish);
    }
};

}

#endif //ORION_SWISH_HPP
