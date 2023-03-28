//
// Created by Raymond on 1/19/23.
//

#ifndef FLARE_LEAKY_RELU_HPP
#define FLARE_LEAKY_RELU_HPP

#include "activation.hpp"

namespace fl
{

template<int TensorRank>
class LeakyReLU : public Activation<Linear, TensorRank>
{
public:
    explicit LeakyReLU(Scalar leak = 0.3);

    void Forward(const Tensor<TensorRank> &inputs) override;

    const Tensor<2> &GetInputGradients2D() override;

    const Tensor<3> &GetInputGradients3D() override;

    const Tensor<4> &GetInputGradients4D() override;

private:
    const Scalar leak = 0.3;

};

} // namespace fl

#include "leaky_relu.ipp"

#endif //FLARE_LEAKY_RELU_HPP
