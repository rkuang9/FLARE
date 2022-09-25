//
// Created by macross on 8/17/22.
//

#ifndef ORION_LINEAR_HPP
#define ORION_LINEAR_HPP

#include "orion/orion_types.hpp"

namespace orion
{

class Linear
{
public:
    static Tensor<2> Activate(const Tensor<2> &features)
    {
        return features;
    }


    static Tensor<3> Activate(const Tensor<3> &features)
    {
        return features;
    }


    static Tensor<2> Gradients(const Tensor<2> &features)
    {
        auto one = static_cast<Scalar>(1.0);
        return Tensor<2>(features.dimensions()).setConstant(one);
    }


    static Tensor<3> Gradients(const Tensor<3> &features)
    {
        auto one = static_cast<Scalar>(1.0);
        return Tensor<2>(features.dimensions()).setConstant(one);
    }
};

}

#endif //ORION_LINEAR_HPP