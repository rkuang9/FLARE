//
// Created by Macross on 12/17/22.
//

#include "conv2dtranspose.hpp"

namespace fl
{

template<typename Activation, int Threads>
Conv2DTranspose<Activation, Threads>::Conv2DTranspose(
        int num_filters, int input_channels, const Kernel &kernel,
        const Stride &stride, const Dilation &dilation, Padding padding,
        const Initializer<4> &initializer):
        Conv2D<Activation, Threads>(num_filters, input_channels, kernel,
                                    stride, dilation, padding, initializer),
        output_padding(padding == Padding::PADDING_SAME ?
                       Dims<2>(stride[0] - 1, stride[1] - 1) : Dims<2>(0, 0))
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
        int num_filters, int input_channels, const Kernel &kernel,
        const Stride &stride, const Dilation &dilation, Padding padding,
        const Dims<2> &output_padding, const Initializer<4> &initializer) :
        Conv2D<Activation, Threads>(num_filters, input_channels, kernel,
                                    stride, dilation, padding, initializer),
        output_padding(output_padding)
{
    if (output_padding[0] >= stride[0] || output_padding[1] >= stride[1] ||
        output_padding[0] < 0 || output_padding[1] < 0) {
        throw std::invalid_argument(
                "OUTPUT PADDING MUST BE >= 0 AND LESS THAN STRIDE");
    }

    if (stride.TotalSize() > 1 && dilation.TotalSize() > 1) {
        throw std::invalid_argument(
                "Stride and dilation may not both be greater than 1");
    }

    this->dL_dk.resize(this->kernels.dimensions());
    this->name = "conv2d_transpose";
}


template<typename Activation, int Threads>
void Conv2DTranspose<Activation, Threads>::Forward(const Tensor<4> &inputs)
{
    fl_assert(inputs.dimension(3) == this->kernels.dimensions().back(),
              "Conv2D::Forward EXPECTED A TENSOR WITH "
                         << this->kernels.dimensions().back() << "CHANNELS" <<
                         ", INSTEAD GOT " << inputs.dimensions().back());

    this->X.resize(inputs.dimensions());
    this->X.device(this->device) = inputs;

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

    // cache paddings for backward propagation on kernel to reuse
    this->fwd_pad[0] = padding_top;
    this->fwd_pad[1] = padding_bottom;
    this->fwd_pad[2] = padding_left;
    this->fwd_pad[3] = padding_right;

    this->Z.resize(inputs.dimension(0), output_h, output_w,
                   this->kernels.dimension(0));
    this->A.resize(this->Z.dimensions());

    this->Z.template device(this->device) = this->ConvolutionForward(
            inputs, this->kernels.reverse(Dims<4, bool>(false, true, true, false)),
            Stride(1, 1), this->dilation, this->stride,
            this->Z.dimensions(),
            padding_top, padding_bottom, padding_left, padding_right);
    this->A.template device(this->device) = Activation::Activate(this->Z);
}


template<typename Activation, int Threads>
void Conv2DTranspose<Activation, Threads>::Backward(const Tensor<4> &gradients)
{
    fl_assert(gradients.dimensions() == this->Z.dimensions(),
              "Conv2DTranspose::Backward EXPECTED GRADIENT DIMS "
                         << this->Z.dimensions() << ", GOT "
                         << gradients.dimensions() << " INSTEAD");

    this->dL_dZ.resize(this->Z.dimensions());
    this->dL_dZ.template device(this->device) =
            gradients * Activation::Gradients(this->Z);

    this->dL_dk.template device(this->device) =
            this->ConvolutionBackwardKernel(
                            this->X, this->dL_dZ,
                            this->dilation, Inflate(1, 1), this->stride,
                            this->kernels.dimensions(),
                            this->fwd_pad[0], this->fwd_pad[1],
                            this->fwd_pad[2], this->fwd_pad[3])
                    .template reverse(Dims<4, bool>(false, true, true, false));
}


template<typename Activation, int Threads>
const Tensor<4> &Conv2DTranspose<Activation, Threads>::GetInputGradients4D()
{
    Eigen::Index pad_h = 0;
    Eigen::Index pad_w = 0;

    if (this->padding == Padding::PADDING_SAME) {
        // from https://github.com/keras-team/keras/blob/ab02bd5c8d75a9c8cc9f78cc7cdb5e7a01307588/keras/utils/conv_utils.py#L208
        pad_h = std::floor((this->dilation[0] * (this->kernel_dim[0] - 1) + 1) / 2);
        pad_w = std::floor((this->dilation[1] * (this->kernel_dim[1] - 1) + 1) / 2);
    }

    // TensorFlow padding calculations, all left side variable names are preserved,
    // right side variables localized to this class's variables as needed
    const Eigen::Index inputRows = this->dL_dZ.dimension(1);
    const Eigen::Index inputCols = this->dL_dZ.dimension(2);

    const Eigen::Index outputRows = this->X.dimension(1);
    const Eigen::Index outputCols = this->X.dimension(2);

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
    // END TensorFlow padding calculations

    const Eigen::Index forward_pad_bottom = 2 * pad_h - forward_pad_top;
    const Eigen::Index forward_pad_right = 2 * pad_w - forward_pad_left;

    this->dL_dX.resize(this->X.dimensions());

    this->dL_dX.template device(this->device) =
            Conv2D<Activation, Threads>::ConvolutionBackwardInput(
                    this->dL_dZ, this->kernels.template reverse(Dims<4, bool>(false, true, true, false)),
                    this->stride, this->dilation, Inflate(1, 1),
                    this->X.dimensions(),
                    forward_pad_top, forward_pad_bottom, forward_pad_left,
                    forward_pad_right);
    return this->dL_dX;
}

} // namespace fl