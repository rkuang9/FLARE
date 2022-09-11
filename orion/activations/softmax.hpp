//
// Created by macross on 8/8/22.
//

#ifndef ORION_SOFTMAX_HPP
#define ORION_SOFTMAX_HPP

#include "orion/orion_types.hpp"

namespace orion
{

class Softmax
{
public:
    static Tensor2D Activate(const Tensor2D &Z)
    {
        return Z.exp() / Z.exp().sum();
    }


    static Tensor2D Gradients(const Tensor2D &Z)
    {
        return (Z >= 0).select(Z.constant(1.0), Z.constant(0.0));
    }
};

} // namespace orion

#endif //ORION_SOFTMAX_HPP
