//
// Created by macross on 8/7/22.
//

#ifndef ORION_DENSE_HPP
#define ORION_DENSE_HPP

#include "layer.hpp"

namespace orion
{

template<typename Activation = Linear>
class Dense : public Layer
{
public:
    explicit Dense(int inputs, int outputs, bool use_bias,
                   const Initializer &initializer = GlorotUniform());

    ~Dense() override = default;

    void Forward(const Tensor<2> &input) override;

    void Forward(const Layer &prev) override;

    // hidden layer back propagation
    void Backward(const Layer &next) override;

    // output layer back propagation
    void Backward(const Loss &loss_function) override;

    void Update(Optimizer &optimizer) override;

    // getter setters
    const Tensor<2> &GetOutput2D() const override;

    const Tensor<2> &GetInputGradients2D() const override;

    Tensor<2> GetGradients() const override;

    const Tensor<2> &GetWeights() const override;

    const Tensor<2> &GetWeightGradients() const override;

    void SetWeights(const Tensor<2> &weights) override;

    Tensor<2> &Bias() override;

    int GetInputRank() const override;

    int GetOutputRank() const override;

    Tensor<2> operator()(const Tensor<2> &tensor) const override;

private:
    // back propagation calculations for hidden and output layers
    void Backward(const Tensor2D &gradients);

    bool use_bias = true;

    Tensor<2> X; // layer input matrix, stacked column-wise
    Tensor<2> Z; // weighted input matrix, Z = W*X + b
    Tensor<2> A; // activated input matrix after applying g(Z)
    Tensor<2> dL_dZ; // derivative of loss w.r.t. inputs, same dim as X, Z, and A

    Tensor<2> w; // weights matrix
    Tensor<2> b; // bias vector
    Tensor<2> dL_dw; // derivative of loss w.r.t. weights
    Tensor<2> dL_db; // derivative of loss w.r.t. bias
};

} // namespace orion

#include "dense.ipp"

#endif //ORION_DENSE_HPP
