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
    Conv2D(int num_filters, const Input &input, const Kernel &kernel,
            //Padding padding = Padding::PADDING_VALID,
           int padding,
           const Stride &stride = Stride(1, 1),
           const Initializer<4> &initializer = GlorotUniform<4>(),
           const Dilation &dilation = Dilation(1, 1));

    ~Conv2D() override = default;

    /**
     * Uses the im2col method to convolve an input image
     * @param inputs
     */
    void Forward(const Tensor<4> &inputs) override;

    void Backward(const LossFunction &loss_function) override;

    void Forward(const Layer &prev) override;

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

private:
    Tensor<4> X;
    Tensor<4> Z;
    Tensor<4> A;
    Tensor<4> dL_dZ;

    Tensor<4> kernels; // weights, NHWC format, N = num filters
    Tensor<4> b; // bias
    Tensor<4> dL_dk;
    Tensor<4> dL_db;

    // the following are calculated once beforehand for the convolution operation
    Eigen::PaddingType padding;
    Eigen::Index padding_w = 0;
    Eigen::Index padding_h = 0;

    Input input_dim;
    Kernel kernel_dim;
    Stride stride_dim;
    Dilation dilation_dim;
    Dims<3> output_dim;

    int num_filters = 0; // TODO: may not be needed
    Eigen::Index num_patches = 0;
    Eigen::Index kernel_size = 0;
};

} // namespace orion

#include "conv2d.ipp"

#endif //ORION_CONV2D_HPP
