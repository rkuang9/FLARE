//
// Created by macross on 10/10/22.
//

#ifndef ORION_MAXPOOLING2D_HPP
#define ORION_MAXPOOLING2D_HPP

#include "layer.hpp"

namespace orion
{

class MaxPooling2D : public Layer
{
public:
    MaxPooling2D(const PoolSize &pool, const Stride &stride,
                 const Dilation &dilation, Padding padding);

    ~MaxPooling2D() override = default;

    void Forward(const Tensor<4> &inputs) override;

    void Forward(const Layer &prev) override;

    void Backward(const LossFunction &loss_function) override;

    void Backward(const Layer &next) override;

    const Tensor<4> &GetOutput4D() const override;

    const Tensor<4> &GetInputGradients4D() const override;

    int GetInputRank() const override;

    int GetOutputRank() const override;

private:
    void Backward(const Tensor<4> &gradients);

    Tensor<4> X;
    Tensor<4> Z;
    Tensor<4> dL_dX;
    Tensor<4> dL_dZ;

    PoolSize pool;
    Stride stride;
    Dilation dilation;
    Padding padding;

};

} // orion

#endif //ORION_MAXPOOLING2D_HPP
