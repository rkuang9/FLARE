//
// Created by macross on 9/1/22.
//

#include "conv2d.hpp"

namespace orion
{

template<typename Activation>
Conv2D<Activation>::Conv2D(int num_filters, const Input &input, const Kernel &kernel,
                           int padding,//Padding padding,
                           const Stride &stride,
                           const Initializer<4> &initializer,
                           const Dilation &dilation) :
        num_filters(num_filters),
        input_dim(input), kernel_dim(kernel),
        stride_dim(stride), dilation_dim(dilation),
        padding(padding == 1 ? Eigen::PADDING_VALID : Eigen::PADDING_SAME)
{
#ifdef ORION_COLMAJOR
    throw std::invalid_argument(
            "Conv2D only supports the NHWC tensor format, use ORION_ROWMAJOR instead");
#endif

    orion_assert(kernel.size() == 2 && stride.size() == 2 && dilation.size() == 2,
                 "kernel, stride, and dilation dimensions require 2 values");

    auto receptive_field_size = kernel[0] * kernel[1];
    auto input_channels = input.back();

    // https://stackoverflow.com/questions/42670274/how-to-calculate-fan-in-and-fan-out-in-xavier-initialization-for-neural-networks
    this->kernels = initializer.
            Initialize(Dims<4>(num_filters, kernel[0], kernel[1], input.back()),
                       static_cast<int>(input_channels * receptive_field_size),
                       static_cast<int>(num_filters * receptive_field_size));

    // set up dimension values required for Im2col convolution
    if (padding == Padding::PADDING_SAME) {
        // required during forward propagation for Im2col patch count
        this->padding_h = (kernel[0] - 1) / 2;
        this->padding_w = (kernel[1] - 1) / 2;
    }

    // use formula ((n + 2p - f) / s) + 1 to compute output dimensions
    Eigen::Index output_h =
            ((this->input_dim[0] + 2 * this->padding_h - kernel_dim[0]) /
             this->stride_dim[0]) + 1;
    Eigen::Index output_w =
            ((this->input_dim[1] + 2 * this->padding_w - kernel_dim[1]) /
             this->stride_dim[1]) + 1;
    this->output_dim = Dims<3>(output_h, output_w, num_filters);

    this->num_patches = this->output_dim[0] * this->output_dim[1];
    this->kernel_size = this->kernel_dim.TotalSize() * this->input_dim.back();
}


// https://stackoverflow.com/questions/55532819/replicating-tensorflows-conv2d-operating-using-eigen-tensors
template<typename Activation>
void Conv2D<Activation>::Forward(const Tensor<4> &inputs)
{
    // if ROW_MAJOR, inputs expected to be NHWC
    this->X = inputs;

    // inputs should have dimensions NHWC
    // (batch, height, width, channels, technically NWHC)
    // while batch N varies, HWC should not change
    auto batches = inputs.dimension(0);

    this->A = inputs
            .extract_image_patches(
                    this->kernel_dim[0], this->kernel_dim[1],
                    this->stride_dim[0], this->stride_dim[1],
                    this->dilation_dim[0], this->dilation_dim[1], this->padding)
            .reshape(Dims<3>(batches, this->num_patches, this->kernel_size))
            .contract(this->kernels.reshape(
                              Dims<2>(this->num_filters, this->kernel_size)),
                      ContractDim{Axes(2, 1)})
            .reshape(Dims<4>(batches, this->output_dim[0], this->output_dim[1],
                             this->num_filters));
}


template<typename Activation>
void Conv2D<Activation>::Backward(const LossFunction &loss_function)
{
    // https://www.youtube.com/watch?v=Pn7RK7tofPg
    Layer::Backward(loss_function);
}


template<typename Activation>
void Conv2D<Activation>::Forward(const Layer &prev)
{
    Layer::Forward(prev);
}


template<typename Activation>
void Conv2D<Activation>::Backward(const Layer &next)
{
    Layer::Backward(next);
}


template<typename Activation>
void Conv2D<Activation>::Update(Optimizer &optimizer)
{
    Layer::Update(optimizer);
}


template<typename Activation>
const Tensor<4> &Conv2D<Activation>::GetOutput4D() const
{
    return this->A;
}


template<typename Activation>
const Tensor<4> &Conv2D<Activation>::GetInputGradients4D() const
{
    return this->dL_dZ;
}


template<typename Activation>
const Tensor<4> &Conv2D<Activation>::GetWeightGradients4D() const
{
    return this->dL_dk;
}


template<typename Activation>
const Tensor<4> &Conv2D<Activation>::GetWeights4D() const
{
    return this->kernels;
}


template<typename Activation>
void Conv2D<Activation>::SetWeights(const Tensor<4> &weights)
{
    if (weights.dimensions() != this->kernels.dimensions()) {
        std::ostringstream error_msg;
        error_msg << this->name << "Conv2D::SetWeights EXPECTED DIMENSIONS "
                << this->kernels.dimensions() << ", GOT " << weights.dimensions();
        throw std::invalid_argument(error_msg.str());
    }

    this->kernels = weights;
}


template<typename Activation>
void Conv2D<Activation>::SetBias(const Tensor<4> &bias)
{
    if (bias.dimensions() != this->b.dimensions()) {
        std::ostringstream error_msg;
        error_msg << this->name << "Conv2D::SetWeights EXPECTED DIMENSIONS "
                << this->b.dimensions() << ", GOT " << bias.dimensions();
        throw std::invalid_argument(error_msg.str());
    }

    this->b = bias;
}


template<typename Activation>
int Conv2D<Activation>::GetInputRank() const
{
    return 4;
}


template<typename Activation>
int Conv2D<Activation>::GetOutputRank() const
{
    return 4;
}


template<typename Activation>
Tensor<4>
Conv2D<Activation>::Convolve(const Tensor<4> &input, const Tensor<4> &kernels,
                             Eigen::Index stride_h, Eigen::Index stride_w,
                             Eigen::Index dilation_h, Eigen::Index dilation_w,
                             Eigen::PaddingType padding,
                             Eigen::Index num_patches,
                             Eigen::Index output_h, Eigen::Index output_w)
{
    Eigen::Index batch_size = input.dimension(0);
    Eigen::Index num_filters = kernels.dimension(0); // the N in kernels' NHWC

    Eigen::Index kernel_h = kernels.dimension(1);
    Eigen::Index kernel_w = kernels.dimension(2);
    Eigen::Index kernel_size = kernel_h * kernel_w * kernels.dimension(3);

    return input
            .extract_image_patches(
                    kernel_h, kernel_w,
                    stride_h, stride_w,
                    dilation_h, dilation_w, padding)
            .reshape(Dims<3>(batch_size, num_patches, kernel_size))
            .contract(
                    kernels.reshape(Dims<2>(num_filters, kernel_size)),
                    ContractDim{Axes(2, 1)})
            .reshape(Dims<4>(batch_size, output_h, output_h, num_filters));
}


} // namespace orion