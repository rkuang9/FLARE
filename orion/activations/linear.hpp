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
    static Tensor2D Activate(const Tensor2D &Z)
    {
        return Z;
    }


    static Tensor2D Gradients(const Tensor2D &Z)
    {
        return Tensor2D(Z.dimensions()).setConstant(1);
    }
};

}

#endif //ORION_LINEAR_HPP