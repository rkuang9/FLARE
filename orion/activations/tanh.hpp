//
// Created by macross on 8/8/22.
//

#ifndef ORION_TANH_HPP
#define ORION_TANH_HPP

#include "orion/orion_types.hpp"

namespace orion {

class TanH
{
public:
    static Tensor<2> Activate(const Tensor<2> &features)
    {
        return features.tanh();
    }


    static Tensor<3> Activate(const Tensor<3> &features)
    {
        return features.tanh();
    }


    static Tensor<2> Gradients(const Tensor<2> &features)
    {
        return 1 - features.tanh().square();
    }


    static Tensor<3> Gradients(const Tensor<3> &features)
    {
        return 1 - features.tanh().square();
    }
};

} // namespace orion

#endif //ORION_TANH_HPP
