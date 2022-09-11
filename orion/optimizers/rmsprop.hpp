//
// Created by macross on 8/17/22.
//

#ifndef ORION_RMSPROP_HPP
#define ORION_RMSPROP_HPP

#include "optimizer.hpp"

namespace orion
{

class RMSprop : public Optimizer
{
public:
    explicit RMSprop(Scalar learning_rate = 0.001, Scalar momentum = 0.9);

    void Minimize(Tensor2D &W, const Tensor2D &dL_dW) override;

    void Minimize(Tensor1D &b, const Tensor1D &dL_db) override;

private:
    Scalar momentum = 0.9;
    Scalar epsilon = 1e-7; // prevent divison by zero

    std::map<const Scalar *, Tensor2D> s_dw;
    std::map<const Scalar *, Tensor1D> s_db;
};

} // orion

#endif //ORION_RMSPROP_HPP
