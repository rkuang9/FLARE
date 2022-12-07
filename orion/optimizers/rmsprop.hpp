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

    void Minimize(Tensor<2> &W, const Tensor<2> &dL_dW) override;

    void Minimize(Tensor<1> &b, const Tensor<1> &dL_db) override;

    void Step() override;

private:
    Scalar momentum = 0.9;
    Scalar epsilon = 1e-7; // prevent divison by zero

    std::map<const Scalar *, Tensor<2>> s_dw;
    std::map<const Scalar *, Tensor<1>> s_db;
};

} // orion

#endif //ORION_RMSPROP_HPP
