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
        Conv2D<Activation, Threads>(num_filters, input, kernel,
                                    stride, dilation, padding, initializer),
        output_padding(Dims<2>(stride[0] - 1, stride[1] - 1))
{
    if (stride.TotalSize() > 1 && dilation.TotalSize() > 1) {
        throw std::invalid_argument(
                "Stride and dilation may not both be greater than 1");
    }

    this->dL_dk.resize(this->kernels.dimensions());
    this->name = "conv2d_transpose";
}


template<typename Activation, int Threads>
Conv2DTranspose<Activation, Threads>::Conv2DTranspose(
        int num_filters, const Input &input, const Kernel &kernel,
        const Stride &stride, const Dilation &dilation, Padding padding,
        const Dims<2> &output_padding, const Initializer<4> &initializer) :
        Conv2D<Activation, Threads>(num_filters, input, kernel,
                                    stride, dilation, padding, initializer),
        output_padding(output_padding)
{
    if (output_padding[0] >= stride[0] || output_padding[1] >= stride[1] ||
        output_padding[0] < 0 || output_padding[1] < 0) {
        throw std::invalid_argument(
                "OUTPUT PADDING MUST BE >= 0 AND LESS THAN STRIDE");
    }
}


template<typename Activation, int Threads>
void Conv2DTranspose<Activation, Threads>::Forward(const Tensor<4> &inputs)
{
    this->X = inputs;
    Eigen::Index input_rows = this->X.dimension(1);
    Eigen::Index input_cols = this->X.dimension(2);

    Eigen::Index output_h;
    Eigen::Index output_w;

    // for calculating output length, not used for actual padding
    Eigen::Index pad_h = 0;
    Eigen::Index pad_w = 0;

    if (this->padding == Padding::PADDING_SAME) {
        // from https://github.com/keras-team/keras/blob/ab02bd5c8d75a9c8cc9f78cc7cdb5e7a01307588/keras/utils/conv_utils.py#L208
        pad_h = std::floor((this->dilation[0] * (this->kernel_dim[0] - 1) + 1) / 2);
        pad_w = std::floor((this->dilation[1] * (this->kernel_dim[1] - 1) + 1) / 2);
    }

    output_h = (input_rows - 1) * this->stride[0] +
               this->dilation[0] * (this->kernel_dim[0] - 1) + 1 - 2 * pad_h +
               this->output_padding[0];
    output_w = (input_cols - 1) * this->stride[1] +
               this->dilation[1] * (this->kernel_dim[1] - 1) + 1 - 2 * pad_w +
               this->output_padding[1];

    // TensorFlow padding calculations, all left side variable names are preserved,
    // right side variables localized to this class's variables as needed
    const Eigen::Index inputRows = output_h;
    const Eigen::Index inputCols = output_w;

    const Eigen::Index outputRows = input_rows;
    const Eigen::Index outputCols = input_cols;

    // Number of filters to apply. This is the same as the output depth of the result
    //const Eigen::Index kernelFilters = this->kernels.dimension(3);
    // Number of channels. This is the same as the input depth.
    //const Eigen::Index kernelChannels = this->kernels.dimension(0);
    const Eigen::Index kernelRows = this->kernels.dimension(1);
    const Eigen::Index kernelCols = this->kernels.dimension(2);

    // This is the effective kernel size, taking into account the (*_in_stride -
    // 1) zero-values
    // inserted between consecutive kernel elements in atrous convolution
    const Eigen::Index kernelRowsEff =
            kernelRows + (kernelRows - 1) * (this->dilation[0] - 1);
    const Eigen::Index kernelColsEff =
            kernelCols + (kernelCols - 1) * (this->dilation[1] - 1);

    // Computing the forward padding
    const Eigen::Index forward_pad_top =
            Eigen::numext::maxi<Eigen::Index>(
                    0, ((outputRows - 1) * this->stride[0] + kernelRowsEff -
                        inputRows) / 2);
    const Eigen::Index forward_pad_left =
            Eigen::numext::maxi<Eigen::Index>(
                    0, ((outputCols - 1) * this->stride[1] + kernelColsEff -
                        inputCols) / 2);
    const Eigen::Index padding_top = kernelRowsEff - 1 - forward_pad_top;
    const Eigen::Index padding_left = kernelColsEff - 1 - forward_pad_left;

    const Eigen::Index padding_bottom =
            inputRows - (outputRows - 1) * this->stride[0] -
            2 - padding_top + kernelRowsEff;
    const Eigen::Index padding_right =
            inputCols - (outputCols - 1) * this->stride[1] -
            2 - padding_left + kernelColsEff;
    // END TensorFlow padding calculations

    this->Z.resize(inputs.dimension(0), output_h, output_w,
                   this->kernels.dimension(0));

    // backward propagation to calculate kernel gradients will reuse these paddings
    this->fwd_pad[0] = padding_top;
    this->fwd_pad[1] = padding_bottom;
    this->fwd_pad[2] = padding_left;
    this->fwd_pad[3] = padding_right;

    // ConvolutionBackwardInput with kernels reversed on height/width also works
    this->Z.template device(this->device) =
            Conv2D<Activation, Threads>::ConvolutionForward(
                    inputs, this->kernels,
                    Stride(1, 1), this->dilation, this->stride,
                    this->Z.dimensions(),
                    padding_top, padding_bottom, padding_left, padding_right);
    this->A = Activation::Activate(this->Z);
}


template<typename Activation, int Threads>
void Conv2DTranspose<Activation, Threads>::Backward(const Tensor<4> &gradients)
{
    this->dL_dZ.resize(this->Z);
    this->dL_dZ.template device(this->device) =
            gradients * Activation::Gradients(this->Z);

    this->dL_dk.template device(this->device) =
            Conv2D<Activation, Threads>::ConvolutionBackwardKernel(
                    this->X, this->dL_dZ,
                    this->dilation, Inflate(1, 1), this->stride,
                    this->kernels.dimensions(),
                    this->fwd_pad[0], this->fwd_pad[1],
                    this->fwd_pad[2], this->fwd_pad[3])
                    .template reverse(Dims<4, bool>(false, true, true, false));
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
    // as with Conv2D's forward propagation, valid padding means no padding while
    // same padding requires padding done in Conv2DTranspose's forward
    Eigen::Index padding_top = 0;
    Eigen::Index padding_bottom = 0;
    Eigen::Index padding_left = 0;
    Eigen::Index padding_right = 0;

    if (this->padding == Padding::PADDING_SAME) {
        padding_top = this->fwd_pad[0];
        padding_bottom = this->fwd_pad[1];
        padding_left = this->fwd_pad[2];
        padding_right = this->fwd_pad[3];
    }

    return Conv2D<Activation, Threads>::ConvolutionBackwardInput(
            this->dL_dZ, this->kernels,
            this->stride, this->dilation, Inflate(1, 1),
            this->X.dimensions(),
            padding_top, padding_bottom, padding_left, padding_right);
}

} // namespace orion