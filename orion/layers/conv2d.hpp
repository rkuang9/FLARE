//
// Created by macross on 9/1/22.
//

#ifndef ORION_CONV2D_HPP
#define ORION_CONV2D_HPP

#include "layer.hpp"

namespace orion
{

template<typename Activation>
class Conv2D : public Layer
{

public:
    /**
     * 2D convolutional layer, performs convolution on images
     *
     * @param num_filters   # filters used (a collection of filters make a kernel)
     * @param input         input dimensions, Input(height, width, channels)
     * @param kernel        kernel dimensions, Kernel(height, width)
     * @param stride        stride dimensions, Stride(height, width)
     * @param dilation      dilation dimensions, Dilation(height, width)
     * @param padding       padding type, Padding::PADDING_VALID or Padding::PADDING_SAME
     * @param initializer   method of kernel initialization
     */
    Conv2D(int num_filters, const Input &input, const Kernel &kernel,
           const Stride &stride, const Dilation &dilation, Padding padding,
           const Initializer<4> &initializer = GlorotUniform<4>());

    Conv2D(int num_filters, const Input &input, const Kernel &kernel,
           Padding padding = Padding::PADDING_VALID,
           const Initializer<4> &initializer = GlorotUniform<4>());

    ~Conv2D() override = default;


    /**
     * Forward propagation, convolve the input tensor
     * with this layer's filters using the Im2col technique
     *
     * While training, all inputs should have the same dimensions
     * While predicting, inputs may be larger than training inputs
     *
     * @param inputs   a rank 4 tensor
     */
    void Forward(const Tensor<4> &inputs) override;


    /**
     * Forward propagation for hidden layers, passes the previous layer's
     * output as input for Forward(inputs)
     *
     * @param prev   a reference to the previous layer
     */
    void Forward(const Layer &prev) override;


    /**
     * Backward propagation for output layers, calculates
     * loss gradients w.r.t. Z using the loss gradients recorded
     * in the loss object
     *
     * @param loss_function   a reference to the loss object
     */
    void Backward(const LossFunction &loss_function) override;


    /**
     * Backward propagation for hidden layers, calculates
     * loss gradients w.r.t. Z
     *
     * @param next   a reference to the next layer
     */
    void Backward(const Layer &next) override;


    /**
     * Updates the layer's weights and bias with dL/dk, dL/db using
     * the provided optimizer
     *
     * @param optimizer   a reference to the optimizer object
     */
    void Update(Optimizer &optimizer) override;


    // getters and setters


    /**
     * @return   layer's activation values
     */
    const Tensor<4> &GetOutput4D() const override;


    /**
     * @return   layer's loss gradients w.r.t. pre-activated output (dL / dZ))
     */
    Tensor<4> GetInputGradients4D() const override;


    /**
     * @return   layer's kernels
     */
    const Tensor<4> &GetWeights4D() const override;


    /**
     * @return   loss gradients w.r.t. weights (dL / dk)
     */
    const Tensor<4> &GetWeightGradients4D() const override;


    /**
     * Set layer's weights
     *
     * @param weights   custom weights with dimensions [output units, input units]
     */
    void SetWeights(const Tensor<4> &weights) override;


    void SetBias(const Tensor<4> &bias) override;


    /**
     * @return   expected rank of forward propagation's input tensor
     */
    int GetInputRank() const override;


    /**
     * @return   expected rank of forward propagation's output tensor
     */
    int GetOutputRank() const override;


    /**
     * ConvolutionForward a batch of images with a batch of kernels
     *
     * @param input     input tensor in format NHWC
     * @param kernels   kernels tensor in format NHWC, where N = # filters
     * @param stride    Stride dimensions (h, w)
     * @param dilation  Dilation dimensions (h, w)
     * @param padding   Padding enum, PADDING_VALID or PADDING_SAME
     * @return          Tensor<4> in format NHWC
     */
    static Tensor<4> ConvolutionForward(
            const Tensor<4> &input, const Tensor<4> &kernels,
            const Stride &stride, const Dilation &dilation, Padding padding);

    /**
     * Backpropagation, calculate derivative of loss w.r.t. kernels is the convolution
     * between layer input and output gradients propagated back from the next layer
     *
     * @param layer_input   Layer input tensor in format NHWC
     * @param gradients     Output gradients in format FHWC, plays role of kernel
     * @param stride        Forward propagation's dilation (not stride) dimensions (h, w)
     * @param dilation      Forward propagation's stride (not dilation) dimensions (h, w)
     * @param padding       Padding enum used in forward propagation
     * @param output_dims   Expected dimensions of the resultant dL/dk tensor (same as kernels)
     * @return
     */
    static Tensor<4> ConvolutionBackwardKernel(
            const Tensor<4> &layer_input, const Tensor<4> &gradients,
            const Stride &stride, const Dilation &dilation,
            Padding padding, const Dims<4> &output_dims);

    /**
     * Backpropagation through inputs, derivative of loss w.r.t. inputs is the
     * "full" convolution between output gradients dL/dZ and layer kernels
     *
     * @param gradients     Output gradients in format NHWF (aka NHWC for next layer)
     * @param kernels       Layer kernels in format FHWC
     * @param stride        Forward propagation's dilation (not stride) dimensions (h, w)
     * @param dilation      Forward propagation's stride (not dilation) dimensions (h, w)
     * @param result_dims   Dimensions of the forward propagation input tensor
     * @return
     */
    static Tensor<4> ConvolutionBackwardInput(
            const Tensor<4> &gradients, const Tensor<4> &kernels,
            const Stride &stride, const Dilation &dilation,
            const Dims<4> &result_dims);

private:
    // runs backpropagation, the Backward() override functions feed into this
    void Backward();

    Tensor<4> X; // layer input image
    Tensor<4> Z; // layer input convolved with kernels
    Tensor<4> A; // activated output of layer inout convolved with kernels
    Tensor<4> dL_dZ; // gradients of layer output, received from next layer
    Tensor<4> dL_dX; // gradients of layer input, passed as dL_dZ to previous layer

    Tensor<4> kernels; // weights, NHWC format, N = num filters
    Tensor<4> b; // bias
    Tensor<4> dL_dk; // loss gradients w.r.t. kernels
    Tensor<4> dL_db; // loss gradients w.r.t. bias

    // convolution hyperparameters
    Eigen::PaddingType padding;
    Input input_dim;
    Kernel kernel_dim;
    Stride stride_dim;
    Dilation dilation_dim;

};

} // namespace orion

#include "conv2d.ipp"

#endif //ORION_CONV2D_HPP
