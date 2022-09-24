//
// Created by macross on 8/7/22.
//

#ifndef ORION_SIGMOID_HPP
#define ORION_SIGMOID_HPP

#include "orion/orion_types.hpp"

namespace orion {

class Sigmoid
{
public:
    static Tensor<2> Activate(const Tensor<2> &features)
    {
        return features.sigmoid();
    }


    static Tensor<3> Activate(const Tensor<3> &features)
    {
        return features.sigmoid();
    }


    static Tensor<2> Gradients(const Tensor<2> &features)
    {
        return features.sigmoid() * (1 - features.sigmoid());
    }


    static Tensor<3> Gradients(const Tensor<3> &features)
    {
        return features.sigmoid() * (1 - features.sigmoid());
    }
};

} // namespace orion

#endif //ORION_SIGMOID_HPP
