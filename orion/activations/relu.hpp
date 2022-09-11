//
// Created by macross on 8/8/22.
//

#ifndef ORION_RELU_HPP
#define ORION_RELU_HPP

#include "orion/orion_types.hpp"

namespace orion
{

class ReLU
{
public:
    static Tensor2D Activate(const Tensor2D &Z)
    {
#ifdef ORION_FLOAT
        return Z.cwiseMax(0.0f);
#else
        return Z.cwiseMax(0.0);
#endif
    }


    static Tensor2D Gradients(const Tensor2D &Z)
    {
        return (Z >= 0).select(Z.constant(1.0), Z.constant(0.0));
    }
};

} // namespace orion

#endif //ORION_RELU_HPP
