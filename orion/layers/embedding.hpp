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
              const Initializer &initializer = GlorotUniform());

    void Forward(const Tensor<2> &input) override;

    void Backward(const Layer &next) override;

    void Update(Optimizer &optimizer) override;

    const Tensor<3> &GetOutput3D() const override;

    Tensor<2> GetGradients() const override;

    const Tensor<2> &GetWeights() const override;

    void SetWeights(const Tensor<2> &weights) override;

    int GetInputRank() const override;

    int GetOutputRank() const override;

    Tensor<2> operator()(const Tensor<2> &tensor) const override;

protected:
    Tensor<2> X;
    Tensor<3> Z; // output
    Tensor<2> dL_dZ;

    Tensor<2> w; // weights

    int embed_dim;
};

} // namespace orion

#endif //ORION_EMBEDDING_HPP
