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
    static Tensor2D Activate(const Tensor2D &Z)
    {
        return Z.tanh();
    }


    static Tensor2D Gradients(const Tensor2D &Z)
    {
        return 1 - Z.tanh().square();
    }
};

} // namespace orion

#endif //ORION_TANH_HPP
