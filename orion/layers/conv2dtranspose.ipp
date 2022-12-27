//
// Created by Macross on 12/17/22.
//

#include "conv2dtranspose.hpp"

namespace orion
{

template<typename Activation, int Threads>
Conv2DTranspose<Activation, Threads>::Conv2DTranspose(
        int num_filters, const Input &input, const Kernel &kernel,
        const Stride &stride, const Dilation &dilation, Padding padding,
        const Initializer<4> &initializer):
        Conv2D<Activation, Threads>(
                num_filters, input, kernel,
                stride, dilation, padding, initializer)
{
    if (stride.TotalSize() > 1 && dilation.TotalSize() > 1) {
        throw std::invalid_argument(
                "Stride and dilation may not both be greater than 1");
    }

    if (padding == Eigen::PADDING_SAME) {
        throw std::invalid_argument(
                "Conv2DTranspose same padding is not supported yet");
    }

    this->dL_dk.resize(this->kernels.dimensions());
    this->name = "conv2d_transpose";
}


template<typename Activation, int Threads>
void Conv2DTranspose<Activation, Threads>::Forward(const Tensor<4> &inputs)
{
    this->X = inputs;

    Eigen::Index output_h;
    Eigen::Index output_w;
    Eigen::Index pad_h = 0;
    Eigen::Index pad_w = 0;

    if (this->padding == Eigen::PADDING_VALID) {
        output_h = (this->X.dimension(1) - 1) * this->stride[0] +
                   this->dilation[0] * (this->kernel_dim[0] - 1) + 1;
        output_w = (this->X.dimension(2) - 1) * this->stride[1] +
                   this->dilation[1] * (this->kernel_dim[0] - 1) + 1;

        pad_h = this->dilation[0] * (this->kernel_dim[0] - 1);
        pad_w = this->dilation[1] * (this->kernel_dim[1] - 1);
    }
    else {
        output_h = this->X.dimension(1) * this->stride[0];
        output_w = this->X.dimension(2) * this->stride[1];
    }

    this->Z.resize(inputs.dimension(0), output_h, output_w,
                   this->kernels.dimension(0));

    this->Z.template device(this->device) =
            Conv2D<Activation, Threads>::ConvolutionForward(
                    inputs, this->kernels, Stride(1, 1),
                    this->dilation, this->stride, this->Z.dimensions(),
                    pad_h, pad_h, pad_w, pad_w);

    this->A = Activation::Activate(this->Z);
}


template<typename Activation, int Threads>
void Conv2DTranspose<Activation, Threads>::Backward()
{
    Eigen::Index pad_h = 0;
    Eigen::Index pad_w = 0;

    if (this->padding == Eigen::PADDING_VALID) {
        pad_h = this->dilation[0] * (this->kernel_dim[0] - 1);
        pad_w = this->dilation[1] * (this->kernel_dim[1] - 1);
    }
    else {
        // this portion, PADDING_SAME, does not work yet
        pad_h = (this->kernels.dimension(1) - 1) * this->stride[0] -
                this->X.dimension(1) +
                this->dilation[0] * (this->dL_dZ.dimension(1) - 1) + 1;
        pad_w = (this->kernels.dimension(2) - 1) * this->stride[1] -
                this->X.dimension(2) +
                this->dilation[1] * (this->dL_dZ.dimension(2) - 1) + 1;
    }

    this->dL_dk.template device(this->device) =
            Conv2D<Activation, Threads>::ConvolutionBackwardKernel(
                    this->X, this->dL_dZ,
                    this->dilation, Inflate(1, 1), this->stride,
                    this->kernels.dimensions(),
                    pad_h, pad_h, pad_w, pad_w);
}


template<typename Activation, int Threads>
void Conv2DTranspose<Activation, Threads>::SetWeights(const Tensor<4> &weights)
{
    if (weights.dimensions() != this->kernels.dimensions()) {
        std::ostringstream error_msg;
        error_msg << this->name << " Conv2D::SetWeights EXPECTED DIMENSIONS "
                  << this->kernels.dimensions() << ", GOT " << weights.dimensions();
        throw std::invalid_argument(error_msg.str());
    }

    this->kernels = weights.reverse(Dims<4, bool>(false, true, true, false));
}


template<typename Activation, int Threads>
Tensor<4> Conv2DTranspose<Activation, Threads>::GetInputGradients4D() const
{
    return Conv2D<Activation, Threads>::ConvolutionBackwardInput(
            this->dL_dZ, this->kernels,
            this->stride, this->dilation, Inflate(1, 1),
            this->X.dimensions(),
            0, 0, 0, 0);
}

} // namespace orion