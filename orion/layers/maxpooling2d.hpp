//
// Created by macross on 10/10/22.
//

#ifndef ORION_MAXPOOLING2D_HPP
#define ORION_MAXPOOLING2D_HPP

#include "layer.hpp"

namespace orion
{

template<int PoolRank>
class MaxPooling2D: public Layer
{
public:
    //MaxPooling(Dim)

    ~MaxPooling2D() override = default;

    void Forward(const Tensor<2> &inputs) override;

    void Forward(const Tensor<3> &inputs) override;

    void Forward(const Tensor<4> &inputs) override;

    void Backward(const LossFunction &loss_function) override;

    void Forward(const Layer &prev) override;

    void Backward(const Layer &next) override;

    void Update(Optimizer &optimizer) override;

    const Tensor<2> &GetOutput2D() const override;

    const Tensor<3> &GetOutput3D() const override;

    const Tensor<4> &GetOutput4D() const override;

    const Tensor<2> &GetInputGradients2D() const override;

    const Tensor<3> &GetInputGradients3D() const override;

    const Tensor<4> &GetInputGradients4D() const override;

    const Tensor<2> &GetWeights() const override;

    const Tensor<4> &GetWeights4D() const override;

    const Tensor<2> &GetWeightGradients() const override;

    const Tensor<4> &GetWeightGradients4D() const override;

    void SetWeights(const Tensor<2> &weights) override;

    void SetWeights(const Tensor<4> &weights) override;

    const Tensor<2> &GetBias() const override;

    void SetBias(const Tensor<2> &bias) override;

    void SetBias(const Tensor<4> &bias) override;

    int GetInputRank() const override;

    int GetOutputRank() const override;

};

} // orion

#endif //ORION_MAXPOOLING2D_HPP
