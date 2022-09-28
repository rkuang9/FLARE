//
// Created by macross on 9/1/22.
//

#ifndef ORION_CONV2D_HPP
#define ORION_CONV2D_HPP

#include "layer.hpp"

namespace orion
{

enum class Padding
{

};

/**
 * Uses data format "NCHW" (batch, channels, rows cols)
 */
template<typename Activation>
class Conv2D : public Layer
{
public:
    Conv2D(int filters, int kernel_height, int kernel_width,
           const Tensor<2>::Dimensions &input_dims,
           const Initializer &initializer = GlorotUniform());

    ~Conv2D() override = default;

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
    Tensor<4> X;
    Tensor<4> Z;
    Tensor<4> A;
    Tensor<4> filters;

};

}


#endif //ORION_CONV2D_HPP
