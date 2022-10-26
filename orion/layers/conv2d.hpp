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
     *
     * @param num_filters
     * @param input
     * @param kernel
     * @param padding
     * @param stride
     * @param initializer
     * @param dilation      known as in_stride in Eigen, the number of values to
     *                      skip, similar to stride, but for the patch-kernel dot
     *                      Dilation(2, 2) means to skip (pass over) one value
     */
    Conv2D(int num_filters, const Input &input, const Kernel &kernel,
           Padding padding = Padding::PADDING_VALID,
           const Stride &stride = Stride(1, 1),
           const Initializer<4> &initializer = GlorotUniform<4>(),
           const Dilation &dilation = Dilation(1, 1));

    ~Conv2D() override = default;

    /**
     * Forward propagation for hidden layers, convolve the input tensor
     * with this layer's filters using the Im2col technique
     *
     * While training, all inputs should have the same dimensions
     * While predicting, inputs may be larger than training inputs
     *
     * @param inputs   a rank 4 tensor
     */
    void Forward(const Tensor<4> &inputs) override;

    void Forward(const Layer &prev) override;

    void Backward(const LossFunction &loss_function) override;

    void Backward(const Layer &next) override;

    void Update(Optimizer &optimizer) override;

    const Tensor<4> &GetOutput4D() const override;

    const Tensor<4> &GetInputGradients4D() const override;

    const Tensor<4> &GetWeightGradients4D() const override;

    const Tensor<4> &GetWeights4D() const override;

    void SetWeights(const Tensor<4> &weights) override;

    void SetBias(const Tensor<4> &bias) override;

    int GetInputRank() const override;

    int GetOutputRank() const override;

    /**
     * Convolve a batch of images with a batch of kernels
     * @param input     Tensor<4> in format NHWC
     * @param kernels   Tensor<4> in format NHWC, where N = # filters
     * @param stride    Stride dimensions (h, w)
     * @param dilation  Dilation dimensions (h, w)
     * @param padding   Padding enum, PADDING_VALID or PADDING_SAME
     * @return          Tensor<4> in format NHWC
     */
    static Tensor<4> Convolve(const Tensor<4> &input, const Tensor<4> &kernels,
                              const Stride &stride, Padding padding,
                              const Dilation &dilation = Dilation(1, 1));

    /**
     * Backpropagation, calculate derivative of loss w.r.t. kernels is the convolution
     * between layer input and output gradients propagated back from the next layer
     *
     * @param layer_input   Layer input tensor in format NHWC
     * @param gradients     Output gradients with same dims as layer output
     * @param stride        Forward propagation's dilation (not stride) dimensions (h, w)
     * @param dilation      Forward propagation's stride (not dilation) dimensions (h, w)
     * @param padding       Padding enum used in forward propagation
     * @param output_dims   Dimensions of the resultant dL/dk, same as kernels
     * @return
     */
    static Tensor<4> ConvolutionBackwardKernel(
            const Tensor<4> &layer_input, const Tensor<4> &gradients,
            const Stride &stride, const Dilation &dilation,
            Padding padding, Dims<4> output_dims);

private:
    void Backward();

    Tensor<4> X;
    Tensor<4> Z;
    Tensor<4> A;
    Tensor<4> dL_dZ;
    Tensor<4> dL_dX; // to be passed on as dL_dZ to previous layer

    Tensor<4> kernels; // weights, NHWC format, N = num filters
    Tensor<4> b; // bias
    Tensor<4> dL_dk;
    Tensor<4> dL_db;

    // the following are calculated once beforehand for the convolution operation
    Eigen::PaddingType padding;


    Input input_dim;
    Kernel kernel_dim;
    Stride stride_dim;
    Dilation dilation_dim;
    Dims<3> output_dim;

};

} // namespace orion

#include "conv2d.ipp"

#endif //ORION_CONV2D_HPP
