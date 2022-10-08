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
    Conv2D(int filters, Kernel kernel, Stride stride,
           Dilation dilation, PaddingType padding = PaddingType::PADDING_VALID,
           const Initializer<2> &initializer = GlorotUniform<2>());

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

    const Tensor<3> &GetWeightGradients3D() const override;

    void SetWeights(const Tensor<3> &weights) override;

    void SetBias(const Tensor<3> &bias) override;

    int GetInputRank() const override;

    int GetOutputRank() const override;

private:
    /**
     * Rotate image tensor clockwise 90 degrees, necessary for
     * extract_image_patches() which expects NWHC format
     *
     * @param tensor   a tensor in format NHWC
     * @return Tensor<4>
     */
    static auto NHWCToNWHC(const Tensor<4> &tensor);

    /**
     * Rotate image tensor counter-clockwise 90 degrees
     * @param tensor   a tensor in format NWHC
     * @return Tensor<4>
     */
    static auto NWHCToNHWC(const Tensor<4> &tensor);

    Tensor<4> X;
    Tensor<4> Z;
    Tensor<4> A;
    Tensor<4> filters;

    Eigen::Index padding_w = 0;
    Eigen::Index padding_h = 0;


    PaddingType padding;
    Kernel kernel;
    Stride stride;
    Dilation dilation;
};

}


#endif //ORION_CONV2D_HPP
