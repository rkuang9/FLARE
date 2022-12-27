//
// Created by Macross on 12/17/22.
//

#ifndef ORION_CONV2DTRANSPOSE_H
#define ORION_CONV2DTRANSPOSE_H

#include "conv2d.hpp"

namespace orion
{

template<typename Activation, int Threads = 2>
class Conv2DTranspose : public Conv2D<Activation, Threads>
{
public:
    Conv2DTranspose(int num_filters, const Input &input, const Kernel &kernel,
                    const Stride &stride, const Dilation &dilation, Padding padding,
                    const Initializer<4> &initializer = GlorotUniform<4>());

    void Forward(const Tensor<4> &inputs) override;

    Tensor<4> GetInputGradients4D() const override;

    void SetWeights(const Tensor<4> &weights) override;

public:
    void Backward() final;
};

} // namespace orion

#include "conv2dtranspose.ipp"

#endif //ORION_CONV2DTRANSPOSE_H
