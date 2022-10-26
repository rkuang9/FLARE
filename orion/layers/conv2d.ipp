//
// Created by macross on 9/1/22.
//

#include "conv2d.hpp"
#include <utility> // for std::pair required for padding arguments


namespace orion
{

template<typename Activation>
Conv2D<Activation>::Conv2D(int num_filters, const Input &input, const Kernel &kernel,
                           Padding padding, const Stride &stride,
                           const Initializer<4> &initializer,
                           const Dilation &dilation) :
        input_dim(input), kernel_dim(kernel),
        stride_dim(stride), dilation_dim(dilation),
        padding(padding == 1 ? Eigen::PADDING_VALID : Eigen::PADDING_SAME)
{
#ifdef ORION_COLMAJOR
    throw std::invalid_argument(
            "Conv2D only supports the NHWC tensor format, use ORION_ROWMAJOR instead");
#endif
    if (padding == Padding::PADDING_SAME) {
        throw std::invalid_argument("SAME padding not working YET");
    }

    orion_assert(kernel.size() == 2 && stride.size() == 2 && dilation.size() == 2,
                 "kernel, stride, and dilation dimensions require 2 values");

    orion_assert(kernel[0] % 2 == 1 && kernel[1] % 2 == 1.0,
                 "Conv2D kernel dimensions should be odd numbers");

    auto receptive_field_size = kernel[0] * kernel[1];
    auto input_channels = input.back();

    // https://stackoverflow.com/questions/42670274/how-to-calculate-fan-in-and-fan-out-in-xavier-initialization-for-neural-networks
    this->kernels = initializer.
            Initialize(Dims<4>(num_filters, kernel[0], kernel[1], input.back()),
                       static_cast<int>(input_channels * receptive_field_size),
                       static_cast<int>(num_filters * receptive_field_size));
}


// https://stackoverflow.com/questions/55532819
template<typename Activation>
void Conv2D<Activation>::Forward(const Tensor<4> &inputs)
{
    // TODO: may have to change assertion to check channels only, and maybe change
    // TODO: the constructor signature to accept channels rather than image dimensions
    orion_assert(/*inputs.dimension(1) == this->input_dim[0] &&
                 inputs.dimension(2) == this->input_dim[1] &&*/
            inputs.dimension(3) == this->input_dim[2],
            "Conv2D::Forward EXPECTED INPUT DIMENSIONS "
                    << Dims<4>(inputs.dimension(0), this->input_dim[0],
                               this->input_dim[1], this->input_dim[2])
                    << " , INSTEAD GOT " << inputs.dimensions());

    this->X = inputs;
    this->Z = Conv2D::Convolve(inputs, this->kernels, this->stride_dim,
                               this->padding, this->dilation_dim);
    this->A = Activation::Activate(this->Z);
}


template<typename Activation>
void Conv2D<Activation>::Forward(const Layer &prev)
{
    this->Forward(prev.GetOutput4D());
}


template<typename Activation>
void Conv2D<Activation>::Backward(const LossFunction &loss_function)
{
    // divide by number of output units of a single batch
    this->dL_dZ = loss_function.GetGradients4D() * Activation::Gradients(this->Z) /
                  static_cast<Scalar>(this->Z.dimensions().TotalSize() /
                                      this->Z.dimension(0));
    this->Backward();
}


template<typename Activation>
void Conv2D<Activation>::Backward(const Layer &next)
{
    this->dL_dZ = next.GetInputGradients4D() * Activation::Gradients(this->Z);
    this->Backward();
}


template<typename Activation>
void Conv2D<Activation>::Backward()
{
    this->dL_dk = Conv2D::ConvolutionBackwardKernel(
            this->X, this->dL_dZ,
            this->dilation_dim, this->stride_dim,
            this->padding, this->kernels.dimensions());

    orion_assert(this->dL_dk.dimensions() == this->kernels.dimensions(),
                 "Conv2D::Backward EXPECTED KERNEL GRADIENTS DIMENSIONS "
                         << this->kernels.dimensions() << ", GOT "
                         << this->dL_dk.dimensions());

    // pad height and width of output gradients, both in each direction once
    Eigen::array<std::pair<int, int>, 4> single_pad_hw;
    single_pad_hw[0] = std::make_pair(0, 0);
    single_pad_hw[1] = std::make_pair(1, 1);
    single_pad_hw[2] = std::make_pair(1, 1);
    single_pad_hw[3] = std::make_pair(0, 0);

    // pad(dL_dZ) * rotate180HW(kernels)
    /*this->dL_dX = Conv2D::Convolve(this->dL_dZ.pad(single_pad_hw),
                                   this->kernels.template reverse(
                                           Dims<4, bool>(false, true, true, false)),
                                   this->stride_dim,
                                   this->padding, this->dilation_dim);*/
}


template<typename Activation>
void Conv2D<Activation>::Update(Optimizer &optimizer)
{
    optimizer.Minimize(this->kernels, this->dL_dk);
}


template<typename Activation>
const Tensor<4> &Conv2D<Activation>::GetOutput4D() const
{
    return this->A;
}


template<typename Activation>
const Tensor<4> &Conv2D<Activation>::GetInputGradients4D() const
{
    return this->dL_dX;
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
        error_msg << this->name << " Conv2D::SetWeights EXPECTED DIMENSIONS "
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
Tensor<4> Conv2D<Activation>::Convolve(const Tensor<4> &input,
                                       const Tensor<4> &kernels,
                                       const Stride &stride, Padding padding,
                                       const Dilation &dilation)
{
    Eigen::Index num_kernels = kernels.dimension(0); // the N in kernels' NHWC
    Eigen::Index kernel_h = kernels.dimension(1);
    Eigen::Index kernel_w = kernels.dimension(2);
    Eigen::Index kernel_size = kernel_h * kernel_w * kernels.dimension(3);

    Eigen::Index padding_h =
            padding == Padding::PADDING_VALID ? 0 : (kernels.dimension(1) - 1) / 2;
    Eigen::Index padding_w =
            padding == Padding::PADDING_VALID ? 0 : (kernels.dimension(2) - 1) / 2;

    Eigen::Index output_h =
            ((input.dimension(1) + 2 * padding_h - dilation[0] * (kernel_h - 1) -
              1) / stride[0]) + 1;
    Eigen::Index output_w =
            ((input.dimension(2) + 2 * padding_w - dilation[1] * (kernel_w - 1) -
              1) / stride[1]) + 1;

    Eigen::Index batch_size = input.dimension(0);
    Eigen::Index num_patches = output_h * output_w;


    Tensor<5> patches = input.extract_image_patches(
            kernel_h, kernel_w,
            stride[0], stride[1],
            dilation[0], dilation[1], padding);

    return input
            .extract_image_patches(
                    kernel_h, kernel_w,
                    stride[0], stride[1],
                    dilation[0], dilation[1], padding)
            .reshape(Dims<3>(batch_size, num_patches, kernel_size))
            .contract(
                    kernels.reshape(Dims<2>(num_kernels, kernel_size)),
                    ContractDim {Axes(2, 1)})
            .reshape(Dims<4>(batch_size, output_h, output_w, num_kernels));
}


template<typename Activation>
Tensor<4> Conv2D<Activation>::ConvolutionBackwardKernel(
        const Tensor<4> &layer_input, const Tensor<4> &gradients,
        const Stride &stride, const Dilation &dilation,
        Padding padding, Dims<4> output_dims)
{
    // layer_input is the layer input 4D tensor in format NHWC
    // gradients is the output gradient 3D tensor in format NHWF, F = # filters
    // gradients serve as the kernel in calculating loss derivative w.r.t. kernels
    // every broadcast must have every value per broadcast dim divided by # times repeated

    Eigen::Index channels = layer_input.dimension(3);
    Eigen::Index batches = gradients.dimension(0);
    Eigen::Index grad_h = gradients.dimension(1);
    Eigen::Index grad_w = gradients.dimension(2);
    Eigen::Index filters = gradients.dimension(3);
    Eigen::Index patches = grad_h * grad_w;
    Eigen::Index grad_size_per_kernel = batches * grad_h * grad_w * channels;

    orion_assert(batches == layer_input.dimension(0),
                 "Conv2D::ConvolutionBackwardKernel EXPECTED GRADIENTS BATCH SIZE "
                         << layer_input.dimension(0) << ", GOT " << batches);

    // reshape gradients via im2col by:
    // 1. reorder dimensions from [batch, height, width, filters] to [filters, batch, height, width]
    //      note: filters is the channels dimension after forward propagation
    // 2. reshape to reintroduce the channels dim, [filters, batch, height, width, 1]
    // 3. broadcast the channels dim to match input channels, [filters, batch, height, width, channels]
    // 4. reshape into im2col [filters, everything else]
    // the division accompanying the broadcast is below at the return value
    auto gradients_im2col = gradients
            .shuffle(Dims<4>(3, 0, 1, 2))
            .reshape(Dims<5>(filters, batches, grad_h, grad_w, 1))
            .broadcast(Dims<5>(1, 1, 1, 1, channels))
            .reshape(Dims<2>(filters, grad_size_per_kernel));

    // reshape layer input via im2col by:
    // 1. extract image patches [batch, patches, grad_h, grad_w, channels], each patch matches gradient dims
    // 2. reorder dimensions from [patches, batch, grad_h, grad_w, channels]
    // 3. reshape into im2col [patches, everything else]
    auto patches_im2col = layer_input
            .extract_image_patches(grad_h, grad_w,
                                   stride[0], stride[1],
                                   dilation[0], dilation[1],
                                   padding)
            .shuffle(Dims<5>(1, 0, 2, 3, 4))
            .reshape(Dims<2>(patches, batches * grad_h * grad_w * channels));

    // convolve the layer input with the backpropagated gradients by:
    // 1. contract the im2col along the [everything else] dim, [filters, patches]
    // 2. reshape patches to actual kernel's height/width, [filters, output_h, output_w, 1]
    // 3. broadcast the channels dim to match actual kernels' channels, [filters, output_h, output_w, channels]
    //      note: output_dims is in the format FHWC, where F = # kernels
    // the resulting tensor's dimensions will match the layer kernel's dimensions
    // since the batches dim was summed during contraction, divide by # batches to get the avg
    // since the channels dim was broadcast # channels times, done twice in total
    // divide by # channels to keep gradients evenly distributed,
    return gradients_im2col
                   .contract(patches_im2col, ContractDim {Axes(1, 1)})
                   .reshape(Dims<4>(filters, output_dims[1], output_dims[2], 1))
                   .broadcast(Dims<4>(1, 1, 1, channels)) /
           static_cast<Scalar>(batches * channels /* * channels*/);
}


} // namespace orion