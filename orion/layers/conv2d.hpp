//
// Created by macross on 9/1/22.
//

#ifndef ORION_CONV2D_HPP
#define ORION_CONV2D_HPP

#include "layer.hpp"

namespace orion
{

template<typename Activation, int threads = 2>
class Conv2D : public Layer
{

public:
    /**
     * 2D convolutional layer, performs convolution on images
     *
     * @param num_filters    num filters to use
     * @param input_channels input tensor channels
     * @param kernel         kernel dimensions, Kernel(height, width)
     * @param stride         stride dimensions, Stride(height, width)
     * @param dilation       dilation dimensions, Dilation(height, width)
     * @param padding        padding type, Padding::PADDING_VALID or Padding::PADDING_SAME
     * @param initializer    kernel initialization method
     */
    Conv2D(int num_filters, int input_channels, const Kernel &kernel,
           const Stride &stride, const Dilation &dilation, Padding padding,
           const Initializer<4> &initializer = GlorotUniform<4>());

    Conv2D(int num_filters, int input_channels, const Kernel &kernel,
           Padding padding = Padding::PADDING_VALID,
           const Initializer<4> &initializer = GlorotUniform<4>());

    ~Conv2D() override = default;

    void Forward(const Tensor<4> &inputs) override;

    void Forward(const Layer &prev) override;

    void Backward(const Tensor<4> &gradients) override;

    void Backward(Layer &next) override;

    void Update(Optimizer &optimizer) override;

    const Tensor<4> &GetOutput4D() const override;

    const Tensor<4> &GetInputGradients4D() override;

    const Tensor<4> &GetWeights4D() const override;

    const Tensor<4> &GetWeightGradients4D() const override;

    void SetWeights(const Tensor<4> &weights) override;

    void SetBias(const Tensor<4> &bias) override;

    int GetInputRank() const override;

    int GetOutputRank() const override;

    void Save(const std::string &path) override;

    void Load(const std::string &path) override;


    /**
     * ConvolutionForward a batch of images with a batch of kernels
     *
     * @param input        input tensor in format NHWC
     * @param kernels      kernels tensor in format NHWC, where N = # filters
     * @param stride       Stride dimensions (h, w)
     * @param dilation     Dilation dimensions (h, w)
     * @param pad_top      input tensor top side padding
     * @param pad_bottom   input tensor bottom side padding
     * @param pad_left     input tensor left side padding
     * @param pad_right    input tensor right side padding
     * @return             Tensor<4> NHWC as tensor op
     */
    EIGEN_STRONG_INLINE
    static auto ConvolutionForward(
            const Tensor<4> &input, const Tensor<4> &kernels,
            const Stride &stride, const Dilation &dilation, const Inflate &inflate,
            const Dims<4> &output_dims,
            Eigen::Index pad_top, Eigen::Index pad_bottom,
            Eigen::Index pad_left, Eigen::Index pad_right);


    /**
     * Backpropagation, calculate derivative of loss w.r.t. kernels is the convolution
     * between layer input and output gradients propagated back from the next layer
     *
     * @param layer_input   Layer input tensor in format NHWC
     * @param gradients     Output gradients in format FHWC, plays role of kernel
     * @param stride        Forward propagation's dilation dimensions (h, w)
     * @param dilation      Forward propagation's stride dimensions (h, w)
     * @param inflate       Dilation on the layer input tensor
     * @param output_dims   Expected dimensions of the resultant dL/dk tensor (same as kernels)
     * @param pad_top       input tensor top side padding
     * @param pad_bottom    input tensor bottom side padding
     * @param pad_left      input tensor left side padding
     * @param pad_right     input tensor right side padding
     * @return              Tensor<4> NHWC as tensor op
     */
    EIGEN_STRONG_INLINE
    static auto ConvolutionBackwardKernel(
            const Tensor<4> &layer_input, const Tensor<4> &gradients,
            const Stride &stride, const Dilation &dilation, const Inflate &inflate,
            const Dims<4> &output_dims,
            Eigen::Index pad_top, Eigen::Index pad_bottom,
            Eigen::Index pad_left, Eigen::Index pad_right);


    /**
     * Backpropagation through inputs, derivative of loss w.r.t. inputs is the
     * "full" convolution between output gradients dL/dZ and layer kernels
     *
     * @param gradients     Output gradients in format NHWF (aka NHWC for next layer)
     * @param kernels       Layer kernels in format FHWC
     * @param stride        Forward propagation's dilation (not stride) dimensions (h, w)
     * @param dilation      Forward propagation's stride (not dilation) dimensions (h, w)
     * @param inflate       Dilation on the gradients tensor
     * @param result_dims   Dimensions of the forward propagation input tensor
     * @param pad_top       gradient tensor top side padding
     * @param pad_bottom    gradient tensor bottom side padding
     * @param pad_left      gradient tensor left side padding
     * @param pad_right     gradient tensor right side padding
     * @return              Tensor<4> NHWC as a tensor op
     */
    EIGEN_STRONG_INLINE
    static auto ConvolutionBackwardInput(
             const Tensor<4> &gradients,  const Tensor<4> &kernels,
             const Stride &stride,  const Dilation &dilation,  const Inflate &inflate,
             const Dims<4> &result_dims,
            Eigen::Index pad_top, Eigen::Index pad_bottom,
            Eigen::Index pad_left, Eigen::Index pad_right);

protected:
    Tensor<4> X; // layer input image, TODO: make this a pointer to avoid copy
    Tensor<4> Z; // layer input convolved with kernels
    Tensor<4> A; // activated output of layer inout convolved with kernels
    Tensor<4> dL_dZ; // gradients of layer output, received from next layer
    Tensor<4> dL_dX; // gradients of layer input

    Tensor<4> kernels; // weights, NHWC format, N = num filters
    Tensor<4> b; // bias
    Tensor<4> dL_dk; // loss gradients w.r.t. kernels
    Tensor<4> dL_db; // loss gradients w.r.t. bias

    // convolution hyperparameters
    const Eigen::PaddingType padding;
    const Kernel kernel_dim;
    const Stride stride;
    const Dilation dilation;

    // multithreading
    Eigen::ThreadPool pool;
    Eigen::ThreadPoolDevice device;

};

} // namespace orion

#include "conv2d.ipp"

#endif //ORION_CONV2D_HPP
