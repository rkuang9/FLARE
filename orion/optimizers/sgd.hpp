//
// Created by macross on 8/8/22.
//

#ifndef ORION_SGD_HPP
#define ORION_SGD_HPP

#include "optimizer.hpp"

namespace orion
{

class SGD : public Optimizer
{
public:
    explicit SGD(Scalar learning_rate = 0.01, Scalar momentum = 0);

    ~SGD() = default;

    void Step() override;

    void Minimize(Tensor<2> &weights, const Tensor<2> &gradients) override;

    void Minimize(Tensor<4> &kernels, const Tensor<4> &gradients) override;

private:
    Scalar momentum = 0; // default 0 means no momentum

    // holds moving averages per layer, stored using Tensor.data() pointer as key
    std::map<const Scalar *, Tensor<2>> v_dw;
    std::map<const Scalar *, Tensor<4>> v_dk;
};

} // namespace orion

#endif //ORION_SGD_HPP
