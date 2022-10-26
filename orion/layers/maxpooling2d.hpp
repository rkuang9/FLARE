//
// Created by macross on 10/10/22.
//

#ifndef ORION_MAXPOOLING2D_HPP
#define ORION_MAXPOOLING2D_HPP

#include "layer.hpp"

namespace orion
{

class MaxPooling2D: public Layer
{
public:
    MaxPooling2D(const PoolSize &pool_dim, const Stride &stride_dim);

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
    Tensor<4> X;
    Tensor<4> Z;

    PoolSize pool_dim;
    Stride stride_dim;

};

} // orion

#endif //ORION_MAXPOOLING2D_HPP
