//
// Created by Macross on 12/17/22.
//

#ifndef FLARE_CONV2DTRANSPOSE_H
#define FLARE_CONV2DTRANSPOSE_H

#include "conv2d.hpp"

namespace fl
{

template<typename Activation, int Threads = 2>
class Conv2DTranspose : public Conv2D<Activation, Threads>
{
public:
    Conv2DTranspose(int num_filters, int input_channels, const Kernel &kernel,
                    const Stride &stride, const Dilation &dilation, Padding padding,
                    const Initializer<4> &initializer = GlorotUniform<4>());

    Conv2DTranspose(int num_filters, int input_channels, const Kernel &kernel,
                    const Stride &stride, const Dilation &dilation, Padding padding,
                    const Dims<2> &output_padding,
                    const Initializer<4> &initializer = GlorotUniform<4>());

    void Forward(const Tensor<4> &inputs) override;

    void Backward(const Tensor<4> &gradients) override;

    const Tensor<4> &GetInputGradients4D() override;

    void SetWeights(const Tensor<4> &weights) override;

private:
    Dims<4> fwd_pad; // save forward top/bottom/left/right padding, reuse in backward

    // Given padding=same, stride=3, input=5, Conv2DTranspose output is 15.
    // However, since output size in the reverse operation, Conv2D, includes an
    // std::floor() call, an input of 13, 14, and 15 can all achieve an output of 5.
    // Output padding controls which of the possible sizes to output
    Dims<2> output_padding;
};

} // namespace fl

#include "conv2dtranspose.ipp"

#endif //FLARE_CONV2DTRANSPOSE_H
