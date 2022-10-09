//
// Created by macross on 9/1/22.
//

#ifndef ORION_EMBEDDING_HPP
#define ORION_EMBEDDING_HPP

#include "layer.hpp"

namespace orion
{

class Embedding : public Layer
{
public:
    Embedding(int vocab_size, int embedding_dim, int input_len,
              const Initializer<2> &initializer = GlorotUniform());

    void Forward(const Tensor<2> &input) override;

    void Backward(const Layer &next) override;

    void Backward(const LossFunction &loss_function) override;

    void Update(Optimizer &optimizer) override;

    const Tensor<3> &GetInputGradients3D() const override;

    const Tensor<3> &GetOutput3D() const override;

    const Tensor<2> &GetWeights() const override;

    const Tensor<2> &GetWeightGradients() const override;

    void SetWeights(const Tensor<2> &weights) override;

    int GetInputRank() const override;

    int GetOutputRank() const override;

    Tensor<2> operator()(const Tensor<2> &tensor) const override;

public:
    void Backward();

    Tensor<2> X;
    Tensor<3> Z; // output (batch, row, col), not related to the Russian military symbol
    Tensor<3> dL_dZ;

    Tensor<2> w; // weights
    Tensor<2> dL_dw; // loss gradients w.r.t. weights

    Eigen::Index embed_dims;
    Eigen::Index input_len;
};

} // namespace orion

#endif //ORION_EMBEDDING_HPP
