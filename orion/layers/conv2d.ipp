//
// Created by macross on 9/1/22.
//

#include "conv2d.hpp"

namespace orion
{

template<typename Activation>
Conv2D<Activation>::Conv2D(
        int num_filters, const Input &input, const Kernel &kernel,
        const Stride &stride, const Dilation &dilation, Padding padding,
        const Initializer<4> &initializer) :
        input_dim(input), kernel_dim(kernel),
        stride_dim(stride), dilation_dim(dilation),
        padding(padding == 1 ? Eigen::PADDING_VALID : Eigen::PADDING_SAME)
{
#ifdef ORION_COLMAJOR
    throw std::invalid_argument(
            "Conv2D only supports the NHWC tensor format, use ORION_ROWMAJOR instead");
#endif

    orion_assert(kernel.size() == 2 && stride.size() == 2 && dilation.size() == 2,
                 "CONV2D KERNEL, STRIDE, AND DILATION DIMS EACH REQUIRE 2 VALUES");

    this->name = "conv2d";


    // initialize kernel values
    // https://stackoverflow.com/questions/42670274/how-to-calculate-fan-in-and-fan-out-in-xavier-initialization-for-neural-networks
    auto receptive_field_size = kernel.height() * kernel.width();
    auto channels = input.back();

    this->kernels = initializer.
            Initialize(Dims<4>(num_filters, kernel.height(),
                               kernel.width(), input.channels()),
                       static_cast<int>(channels * receptive_field_size),
                       static_cast<int>(num_filters * receptive_field_size));
}


template<typename Activation>
void Conv2D<Activation>::Forward(const Tensor<4> &inputs)
{
    orion_assert(inputs.dimension(3) == this->input_dim[2],
                 "Conv2D::Forward EXPECTED INPUT DIMENSIONS "
                         << Dims<4>(inputs.dimension(0), this->input_dim[0],
                                    this->input_dim[1], this->input_dim[2])
                         << " , INSTEAD GOT " << inputs.dimensions());

    this->X = inputs;
    this->Z = Conv2D::ConvolutionForward(
            inputs, this->kernels, this->stride_dim,
            this->dilation_dim, this->padding);
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

    this->dL_dX = Conv2D::ConvolutionBackwardInput(
            this->dL_dZ, this->kernels,
            this->dilation_dim, this->stride_dim, this->X.dimensions());
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
Tensor<4> Conv2D<Activation>::ConvolutionForward(
        const Tensor<4> &input, const Tensor<4> &kernels,
        const Stride &stride, const Dilation &dilation, Padding padding)
{
    Eigen::Index num_kernels = kernels.dimension(0); // the N in kernels' NHWC
    Eigen::Index kernel_h = kernels.dimension(1);
    Eigen::Index kernel_w = kernels.dimension(2);
    Eigen::Index kernel_size = kernel_h * kernel_w * kernels.dimension(3);
    Eigen::Index input_h = input.dimension(1);
    Eigen::Index input_w = input.dimension(2);
    Eigen::Index batch_size = input.dimension(0);

    Eigen::Index output_h = 0;
    Eigen::Index output_w = 0;
    Eigen::Index pad_h = 0;
    Eigen::Index pad_w = 0;

    if (padding == Eigen::PADDING_SAME) {
        // same padding for strided convolutions will have resolution decreased
        output_h = std::ceil(static_cast<Scalar>(input_h) /
                             static_cast<Scalar>(stride.height()));
        output_w = std::ceil(static_cast<Scalar>(input_w) /
                             static_cast<Scalar>(stride.width()));

        // if uneven padding, extra goes to bottom/right
        pad_h = (output_h - 1) * stride.height() -
                input_h + dilation.height() * (kernel_h - 1) + 1;
        pad_w = (output_w - 1) * stride.width() -
                input_w + dilation.width() * (kernel_w - 1) + 1;
    }
    else {
        output_h = 1 + (input_h - dilation.height() * (kernel_h - 1) - 1) /
                       stride.height();
        output_w = 1 + (input_w - dilation.width() * (kernel_w - 1) - 1) /
                       stride.width();
    }

    Eigen::Index num_patches = output_h * output_w;

    return input
            .extract_image_patches(
                    kernel_h, kernel_w,
                    stride.height(), stride.width(),
                    dilation.height(), dilation.width(),
                    1, 1,
                    pad_h / 2, pad_h - pad_h / 2,
                    pad_w / 2, pad_w - pad_w / 2, 0.0)
            .reshape(Dims<3>(batch_size, num_patches, kernel_size))
            .contract(kernels.reshape(Dims<2>(num_kernels, kernel_size)),
                      ContractDim {Axes(2, 1)})
            .reshape(Dims<4>(batch_size, output_h, output_w, num_kernels));
}


template<typename Activation>
Tensor<4> Conv2D<Activation>::ConvolutionBackwardKernel(
        const Tensor<4> &layer_input, const Tensor<4> &gradients,
        const Stride &stride, const Dilation &dilation,
        Padding padding, const Dims<4> &output_dims)
{
    // stride and dilation from forward propagation are swapped in backpropagation
    // layer_input is the layer input 4D tensor in format [N,H,W,C]
    // gradients is the output gradient 4D tensor in format [N,H,W,F], F = # filters
    // gradients play the role of the kernel in this backward dL/dk convolution
    Eigen::Index channels = layer_input.dimension(3);
    Eigen::Index batches = gradients.dimension(0);
    Eigen::Index grad_h = gradients.dimension(1);
    Eigen::Index grad_w = gradients.dimension(2);
    Eigen::Index filters = gradients.dimension(3);
    Eigen::Index patches = output_dims[1] * output_dims[2]; // H * W from [F,H,W,C]
    Eigen::Index size_per_kernel = batches * grad_h * grad_w * channels;

    orion_assert(batches == layer_input.dimension(0),
                 "Conv2D::ConvolutionBackwardKernel EXPECTED GRADIENTS BATCH SIZE "
                         << layer_input.dimension(0) << ", GOT " << batches);

    // reshape gradients to im2col tensor by:
    // 1. reorder dimensions from [N,H,W,F] to [F,N,H,W], F = # kernels
    // 2. reintroduce the channels dim by reshaping to, [F,N,H,W,1]
    // 3. repeat the channels dim to match input channels, [F,N,H,W,C]
    // 4. reshape into 2D tensor [F, grad_size_per_kernel]
    // the division accompanying the broadcast is below at the return value
    auto gradients_im2col = gradients
            .shuffle(Dims<4>(3, 0, 1, 2))
            .reshape(Dims<5>(filters, batches, grad_h, grad_w, 1))
            .broadcast(Dims<5>(1, 1, 1, 1, channels))
            .reshape(Dims<2>(filters, size_per_kernel));

    Eigen::Index pad_h = 0;
    Eigen::Index pad_w = 0;

    if (padding == Eigen::PADDING_SAME) {
        // if uneven padding, extra goes to bottom/right
        pad_h = (output_dims[1] - 1) * stride.height() -
                layer_input.dimension(1) + dilation.height() * (grad_h - 1) + 1;
        pad_w = (output_dims[2] - 1) * stride.width() -
                layer_input.dimension(2) + dilation.width() * (grad_w - 1) + 1;
    }

    // reshape layer input to im2col tensor by:
    // 1. pad the layer input tensor if same padding is used
    // 2. extract image patches [N,P,H,W,C], P = # patches = # times gradients fit the image
    // 3. reorder dimensions from [N,P,H,W,C] to [P,N,H,W,C]
    // 4. reshape into 2D tensor [P, grad_size_per_kernel]
    auto patches_im2col = layer_input
            .extract_image_patches(grad_h, grad_w,
                                   stride[0], stride[1],
                                   dilation[0], dilation[1],
                                   1, 1,
                                   pad_h / 2, pad_h - pad_h / 2,
                                   pad_w / 2, pad_w - pad_w / 2, 0.0)
            .shuffle(Dims<5>(1, 0, 2, 3, 4))
            .reshape(Dims<2>(patches, size_per_kernel));

    // convolve the layer input with the backpropagated gradients by:
    // 1. contract the 2 im2col tensors along the [grad_size_per_kernel] dim to get [F,P]
    // 2. reshape patches dim to actual kernel's height/width, [F, output_h, output_w, 1]
    // 3. repeat the channels dim to match actual kernels' channels, [F, output_h, output_w, C]
    // the resulting tensor's dimensions will match the layer kernel's dimensions
    // since the batch dim N was summed during contraction, divide by # batches to get the avg
    // since the channels dim was broadcast, divide by # times it was done so
    return gradients_im2col
                   .contract(patches_im2col, ContractDim {Axes(1, 1)})
                   .reshape(Dims<4>(filters, output_dims[1], output_dims[2], 1))
                   .broadcast(Dims<4>(1, 1, 1, channels)) /
           static_cast<Scalar>(batches * channels /* * channels*/);
}


template<typename Activation>
Tensor<4> Conv2D<Activation>::ConvolutionBackwardInput(
        const Tensor<4> &gradients, const Tensor<4> &kernels,
        const Stride &stride, const Dilation &dilation,
        const Dims<4> &result_dims)
{
    Eigen::Index batches = gradients.dimension(0);
    Eigen::Index kernel_h = kernels.dimension(1);
    Eigen::Index kernel_w = kernels.dimension(2);
    Eigen::Index channels = kernels.dimension(3);

    Eigen::Index num_kernels = kernels.dimension(0);
    Eigen::Index num_patches = result_dims[1] * result_dims[2];
    Eigen::Index grad_h = gradients.dimension(1);
    Eigen::Index grad_w = gradients.dimension(2);

    Eigen::Index layer_input_h = result_dims[1];
    Eigen::Index layer_input_w = result_dims[2];

    // if uneven padding, extra goes to bottom/right
    Eigen::Index pad_h = (layer_input_h - 1) * stride.height() -
                         grad_h + dilation.height() * (kernel_h - 1) + 1;
    Eigen::Index pad_w = (layer_input_w - 1) * stride.width() -
                         grad_w + dilation.width() * (kernel_w - 1) + 1;

    // reshape the gradients to im2col
    // 1. extract image patches to get a 5D tensor in [N,P,H,W,F] format,
    //    P = #patches, F = #filters from prev layer (or #channels)
    // 2. collapse the patches dimensions to get a 3D tensor in [N,P,H*W*F] format
    auto gradients_im2col = gradients
            .extract_image_patches(kernel_h, kernel_w,
                                   stride.height(), stride.width(),
                                   dilation.height(), dilation.width(),
                                   1, 1,
                                   pad_h / 2, pad_h - pad_h / 2,
                                   pad_w / 2, pad_w - pad_w / 2, 0)
            .reshape(Dims<3>(batches, num_patches,
                             kernel_h * kernel_w * num_kernels));

    // reshape the kernels to im2col
    // 1. backpropagation through inputs requires kernels to be flipped 180 deg
    // 2. reorder dimensions from [F,H,W,C] to [C,H,W,F], F = #filters
    // 3. reshape into a 2D tensor in [C,H*W*F] format
    auto kernels_im2col = kernels
            .reverse(Dims<4, bool>(false, true, true, false))
            .eval()
            .shuffle(Dims<4>(3, 1, 2, 0))
            .reshape(Dims<2>(channels, kernel_h * kernel_w * num_kernels));

    // multiply the gradient and kernel tensors
    // 1. given gradients im2col dims: [N,P,H*W*F] and kernels im2col dims: [C,H*W*F],
    //    contract on the H*W*F dimensions of the gradients and im2col tensors
    // 2. reshape the resulting tensor from [N,P,C] to [N,H,W,C], this dL/dX tensor
    //    is passed back to the previous layer as dL/dZ
    return gradients_im2col
            .contract(kernels_im2col, ContractDim {Axes(2, 1)})
            .reshape(result_dims);
}

} // namespace orion