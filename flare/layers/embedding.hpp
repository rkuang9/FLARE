//
// Created by macross on 9/1/22.
//

#ifndef FLARE_EMBEDDING_HPP
#define FLARE_EMBEDDING_HPP

#include "layer.hpp"

namespace fl
{

class Embedding : public Layer
{
public:
    Embedding(int vocab_size, int embedding_dim, int input_len,
              const Initializer<2> &initializer = GlorotUniform<2>());

    void Forward(const Tensor<2> &input) override;

    void Backward(const Tensor<3> &gradients) override;

    void Backward(Layer &next) override;

    void Update(Optimizer &optimizer) override;

    const Tensor<3> &GetOutput3D() const override;

    const Tensor<2> &GetWeights() const override;

    const Tensor<2> &GetWeightGradients() const override;

    void SetWeights(const Tensor<2> &weights) override;

    int GetInputRank() const override;

    int GetOutputRank() const override;

    void Save(const std::string &path) override;

    void Load(const std::string &path) override;

public:
    void Backward();

    Tensor<2> X;
    Tensor<3> Z; // output (batch, row, col), not related to the Russian military symbol
    Tensor<3> dL_dZ;

    Tensor<2> w; // weights
    Tensor<2> dL_dw; // loss gradients w.r.t. weights

    Eigen::Index embed_dims;
    Eigen::Index input_len;

    // multithreading
    Eigen::ThreadPool pool = Eigen::ThreadPool(
            (int) std::thread::hardware_concurrency());
    Eigen::ThreadPoolDevice device = Eigen::ThreadPoolDevice(&pool, 2);
};

} // namespace fl

#endif //FLARE_EMBEDDING_HPP