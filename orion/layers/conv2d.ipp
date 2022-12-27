//
// Created by macross on 9/1/22.
//

#include "conv2d.hpp"

namespace orion
{

template<typename Activation, int Threads>
Conv2D<Activation, Threads>::Conv2D(
        int num_filters, const Input &input, const Kernel &kernel,
        const Stride &stride, const Dilation &dilation, Padding padding,
        const Initializer<4> &initializer) :
        input_dim(input), kernel_dim(kernel),
        stride(stride), dilation(dilation),
        padding(padding == 1 ? Eigen::PADDING_VALID : Eigen::PADDING_SAME),
        pool((int) std::thread::hardware_concurrency()),
        device(&pool, 2)
{
#ifdef ORION_COLMAJOR
    throw std::invalid_argument(
            "Conv2D only supports the NHWC tensor format, use ORION_ROWMAJOR instead");
#endif
    orion_assert(kernel.size() == 2 && stride.size() == 2 && dilation.size() == 2,
                 "CONV2D KERNEL, STRIDE, AND DILATION DIMS EACH REQUIRE 2 VALUES");

    if (padding != Eigen::PADDING_VALID && padding != Eigen::PADDING_SAME) {
        throw std::invalid_argument(this->name + " invalid padding type");
    }

    // https://stackoverflow.com/questions/42670274/how-to-calculate-fan-in-and-fan-out-in-xavier-initialization-for-neural-networks
    this->kernels = initializer.
            Initialize(
            Dims<4>(num_filters, kernel[0],
                    kernel[1], input[2]),
            static_cast<int>(input.back() * kernel[0] * kernel[1]),
            static_cast<int>(num_filters * kernel[0] * kernel[1]));

    this->dL_dk.resize(this->kernels.dimensions());
    this->name = "conv2d";
}


template<typename Activation, int Threads>
Conv2D<Activation, Threads>::Conv2D(
        int num_filters, const Input &input, const Kernel &kernel,
        Padding padding, const Initializer<4> &initializer)
        : Conv2D(num_filters, input, kernel, Stride(1, 1),
                 Dilation(1, 1), padding, initializer)
{
    // nothing to do
}


template<typename Activation, int Threads>
void Conv2D<Activation, Threads>::Forward(const Tensor<4> &inputs)
{
    orion_assert(inputs.dimension(3) == this->input_dim[2],
                 "Conv2D::Forward EXPECTED INPUT DIMENSIONS "
                         << Dims<4>(inputs.dimension(0), this->input_dim[0],
                                    this->input_dim[1], this->input_dim[2])
                         << " , INSTEAD GOT " << inputs.dimensions());

    this->X = inputs;

    Eigen::Index output_h;
    Eigen::Index output_w;
    Eigen::Index pad_h = 0;
    Eigen::Index pad_w = 0;

    if (this->padding == Eigen::PADDING_VALID) {
        output_h =
                1 + (this->X.dimension(1) -
                     this->dilation[0] * (this->kernel_dim[0] - 1) - 1) /
                    this->stride[0];
        output_w =
                1 + (this->X.dimension(2) -
                     this->dilation[1] * (this->kernel_dim[1] - 1) - 1) /
                    this->stride[1];
    }
    else {
        // same padding for strided convolutions will have resolution decreased
        output_h = std::ceil(static_cast<Scalar>(this->X.dimension(1)) /
                             static_cast<Scalar>(this->stride[0]));
        output_w = std::ceil(static_cast<Scalar>(this->X.dimension(2)) /
                             static_cast<Scalar>(this->stride[1]));

        // if uneven padding, extra goes to bottom/right
        pad_h = (output_h - 1) * this->stride[0] -
                this->X.dimension(1) +
                this->dilation[0] * (this->kernel_dim[0] - 1) + 1;
        pad_w = (output_w - 1) * this->stride[1] -
                this->X.dimension(2) +
                this->dilation[1] * (this->kernel_dim[1] - 1) + 1;
    }

    this->Z.resize(inputs.dimension(0), output_h, output_w,
                   this->kernels.dimension(0));

    this->Z.template device(this->device) = Conv2D::ConvolutionForward(
            inputs, this->kernels, this->stride,
            this->dilation, Inflate(1, 1), this->Z.dimensions(),
            pad_h / 2, pad_h - pad_h / 2, pad_w / 2, pad_w - pad_w / 2);

    this->A = Activation::Activate(this->Z);
}


template<typename Activation, int Threads>
void Conv2D<Activation, Threads>::Forward(const Layer &prev)
{
    this->Forward(prev.GetOutput4D());
}


template<typename Activation, int Threads>
void Conv2D<Activation, Threads>::Backward(const LossFunction &loss_function)
{
    this->dL_dZ = loss_function.GetGradients4D() * Activation::Gradients(this->Z);
    this->Backward();
}


template<typename Activation, int Threads>
void Conv2D<Activation, Threads>::Backward(const Layer &next)
{
    this->dL_dZ.resize(this->Z.dimensions());
    this->dL_dZ.template device(this->device) =
            next.GetInputGradients4D() * Activation::Gradients(this->Z);
    this->Backward();
}


template<typename Activation, int Threads>
void Conv2D<Activation, Threads>::Backward()
{
    // if uneven padding, extra goes to bottom/right, negative padding will remove
    Eigen::Index pad_h =
            (this->kernels.dimension(1) - 1) * this->dilation[0] -
            this->X.dimension(1) +
            this->stride[0] * (this->dL_dZ.dimension(1) - 1) + 1;
    Eigen::Index pad_w =
            (this->kernels.dimension(2) - 1) * this->dilation[1] -
            this->X.dimension(2) +
            this->stride[1] * (this->dL_dZ.dimension(2) - 1) + 1;

    this->dL_dk.template device(this->device) = Conv2D::ConvolutionBackwardKernel(
            this->X, this->dL_dZ,
            this->dilation, this->stride, Inflate(1, 1),
            this->kernels.dimensions(),
            pad_h / 2, pad_h - pad_h / 2, pad_w / 2, pad_w - pad_w / 2);

    orion_assert(this->dL_dk.dimensions() == this->kernels.dimensions(),
                 "Conv2D::Backward EXPECTED KERNEL GRADIENTS DIMENSIONS "
                         << this->kernels.dimensions() << ", GOT "
                         << this->dL_dk.dimensions());
}


template<typename Activation, int Threads>
void Conv2D<Activation, Threads>::Update(Optimizer &optimizer)
{
    optimizer.Minimize(this->kernels, this->dL_dk);
}


template<typename Activation, int Threads>
const Tensor<4> &Conv2D<Activation, Threads>::GetOutput4D() const
{
    return this->A;
}


template<typename Activation, int Threads>
Tensor<4> Conv2D<Activation, Threads>::GetInputGradients4D() const
{
    // the following padding calculations were taken from TensorFlow's SpatialConvolutionBacKWardInput
    const Eigen::Index kernelRowsEff =
            this->kernels.dimension(1) +
            (this->kernels.dimension(1) - 1) * (this->dilation[0] - 1);
    const Eigen::Index kernelColsEff =
            this->kernels.dimension(1) +
            (this->kernels.dimension(1) - 1) * (this->dilation[1] - 1);

    const Eigen::Index outputRows = this->dL_dZ.dimension(1);
    const Eigen::Index outputCols = this->dL_dZ.dimension(2);

    // Computing the forward padding
    const Eigen::Index forward_pad_top = Eigen::numext::maxi<Eigen::Index>(
            0, ((outputRows - 1) * this->stride[0] + kernelRowsEff -
                this->X.dimension(1)) / 2);
    const Eigen::Index forward_pad_left = Eigen::numext::maxi<Eigen::Index>(
            0, ((outputCols - 1) * this->stride[1] + kernelColsEff -
                this->X.dimension(2)) / 2);

    const Eigen::Index padding_top = kernelRowsEff - 1 - forward_pad_top;
    const Eigen::Index padding_left = kernelColsEff - 1 - forward_pad_left;

    const Eigen::Index padding_bottom =
            this->X.dimension(1) - (outputRows - 1) * this->stride[0] -
            2 - padding_top + kernelRowsEff;
    const Eigen::Index padding_right =
            this->X.dimension(2) - (outputCols - 1) * this->stride[1] -
            2 - padding_left + kernelColsEff;

    return Conv2D::ConvolutionBackwardInput(
            this->dL_dZ, this->kernels,
            this->dilation, Dilation(1, 1), this->stride,
            this->X.dimensions(),
            padding_top, padding_bottom, padding_left, padding_right);
}


template<typename Activation, int Threads>
const Tensor<4> &Conv2D<Activation, Threads>::GetWeightGradients4D() const
{
    return this->dL_dk;
}


template<typename Activation, int Threads>
const Tensor<4> &Conv2D<Activation, Threads>::GetWeights4D() const
{
    return this->kernels;
}


template<typename Activation, int Threads>
void Conv2D<Activation, Threads>::SetWeights(const Tensor<4> &weights)
{
    if (weights.dimensions() != this->kernels.dimensions()) {
        std::ostringstream error_msg;
        error_msg << this->name << " Conv2D::SetWeights EXPECTED DIMENSIONS "
                  << this->kernels.dimensions() << ", GOT " << weights.dimensions();
        throw std::invalid_argument(error_msg.str());
    }

    this->kernels = weights;
}


template<typename Activation, int Threads>
void Conv2D<Activation, Threads>::SetBias(const Tensor<4> &bias)
{
    if (bias.dimensions() != this->b.dimensions()) {
        std::ostringstream error_msg;
        error_msg << this->name << "Conv2D::SetWeights EXPECTED DIMENSIONS "
                  << this->b.dimensions() << ", GOT " << bias.dimensions();
        throw std::invalid_argument(error_msg.str());
    }

    this->b = bias;
}


template<typename Activation, int Threads>
int Conv2D<Activation, Threads>::GetInputRank() const
{
    return 4;
}


template<typename Activation, int Threads>
int Conv2D<Activation, Threads>::GetOutputRank() const
{
    return 4;
}


template<typename Activation, int Threads>
auto Conv2D<Activation, Threads>::ConvolutionForward(
        const Tensor<4> &input, const Tensor<4> &kernels,
        const Stride &stride, const Dilation &dilation, const Inflate &inflate,
        const Dims<4> &output_dims,
        Eigen::Index pad_top, Eigen::Index pad_bottom,
        Eigen::Index pad_left, Eigen::Index pad_right)
{
    Eigen::Index num_kernels = kernels.dimension(0);
    Eigen::Index kernel_h = kernels.dimension(1);
    Eigen::Index kernel_w = kernels.dimension(2);
    Eigen::Index kernel_size = kernel_h * kernel_w * kernels.dimension(3);
    Eigen::Index batch_size = input.dimension(0);

    Eigen::Index output_h = output_dims[1];
    Eigen::Index output_w = output_dims[2];

    Eigen::Index num_patches = output_h * output_w;

    return input
            .extract_image_patches(
                    kernel_h, kernel_w,
                    stride[0], stride[1],
                    dilation[0], dilation[1],
                    inflate[0], inflate[1],
                    pad_top, pad_bottom, pad_left, pad_right, 0.0)
            .reshape(Dims<3>(batch_size, num_patches, kernel_size))
            .contract(kernels.reshape(Dims<2>(num_kernels, kernel_size)),
                      ContractDim {Axes(2, 1)})
            .reshape(Dims<4>(batch_size, output_h, output_w, num_kernels));
}


template<typename Activation, int Threads>
auto Conv2D<Activation, Threads>::ConvolutionBackwardKernel(
        const Tensor<4> &layer_input, const Tensor<4> &gradients,
        const Stride &stride, const Dilation &dilation, const Inflate &inflate,
        const Dims<4> &output_dims, Eigen::Index pad_top, Eigen::Index pad_bottom,
        Eigen::Index pad_left, Eigen::Index pad_right)
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
            .eval()
            .reshape(Dims<5>(filters, batches, grad_h, grad_w, 1))
            .broadcast(Dims<5>(1, 1, 1, 1, channels))
            .reshape(Dims<2>(filters, size_per_kernel));

    // reshape layer input to im2col tensor by:
    // 1. pad the layer input tensor if same padding is used
    // 2. extract image patches [N,P,H,W,C], P = # patches = # times gradients fit the image
    // 3. reorder dimensions from [N,P,H,W,C] to [P,N,H,W,C]
    // 4. reshape into 2D tensor [P, grad_size_per_kernel]
    auto patches_im2col = layer_input
            .extract_image_patches(grad_h, grad_w,
                                   stride[0], stride[1],
                                   dilation[0], dilation[1],
                                   inflate[0], inflate[1],
                                   pad_top, pad_bottom, pad_left, pad_right, 0.0)
            .shuffle(Dims<5>(1, 0, 2, 3, 4))
            .eval()
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


template<typename Activation, int Threads>
auto Conv2D<Activation, Threads>::ConvolutionBackwardInput(
        const Tensor<4> &gradients, const Tensor<4> &kernels,
        const Stride &stride, const Dilation &dilation, const Inflate &inflate,
        const Dims<4> &result_dims, Eigen::Index pad_top, Eigen::Index pad_bottom,
        Eigen::Index pad_left, Eigen::Index pad_right)
{
    Eigen::Index batches = gradients.dimension(0);
    Eigen::Index kernel_h = kernels.dimension(1);
    Eigen::Index kernel_w = kernels.dimension(2);
    Eigen::Index channels = kernels.dimension(3);

    Eigen::Index num_kernels = kernels.dimension(0);
    Eigen::Index num_patches = result_dims[1] * result_dims[2];

    // reshape the gradients to im2col
    // 1. extract image patches to get a 5D tensor in [N,P,H,W,F] format,
    //    P = #patches, F = #filters from prev layer (or #channels)
    // 2. collapse the patches dimensions to get a 3D tensor in [N,P,H*W*F] format
    auto gradients_im2col = gradients
            .extract_image_patches(kernel_h, kernel_w,
                                   stride[0], stride[1],
                                   dilation[0], dilation[1],
                                   inflate[0], inflate[1],
                                   pad_top, pad_bottom, pad_left, pad_right, 0)
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
            .eval()
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