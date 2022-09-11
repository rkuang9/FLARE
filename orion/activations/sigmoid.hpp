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
    static Tensor2D Activate(const Tensor2D &Z)
    {
        return Z.sigmoid();
    }


    static Tensor2D Gradients(const Tensor2D &Z)
    {
        return Z.sigmoid() * (1 - Z.sigmoid());
    }
};

} // namespace orion

#endif //ORION_SIGMOID_HPP
