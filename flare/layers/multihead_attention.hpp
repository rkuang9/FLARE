//
// Created by R on 4/23/23.
//

#ifndef FLARE_MULTIHEAD_ATTENTION_HPP
#define FLARE_MULTIHEAD_ATTENTION_HPP

#include "layer.hpp"

namespace fl
{

class MultiHeadAttention : public Layer
{
public:
    /**
     *
     * @param input_dim   number of features in input
     * @param key_dim     size of query and key dimensions per head
     * @param value_dim   size of value dimension per head
     * @param initializer
     */
    MultiHeadAttention(Eigen::Index heads, Eigen::Index input_dim,
                       Eigen::Index query_dim, Eigen::Index value_dim,
                       const Initializer<3> &initializer = GlorotUniform<3>());

    // inputs = [query, key, value]
    void Forward(const std::vector<fl::Tensor<3>> &inputs);

    // self attention where query = key = value
    void Forward(const Tensor<3> &inputs) override;

    void Forward(const Layer &prev) override;

    void Backward(const Tensor<2> &gradients) override;

    void Backward(Layer &next) override;

    void Update(Optimizer &optimizer) override;

    const Tensor<2> &GetOutput2D() const override;

    const Tensor<2> &GetInputGradients2D() override;

    std::vector<fl::Tensor<2>> GetWeights2D() const override;

    std::vector<fl::Tensor<2>> GetWeightGradients2D() const override;

    void SetWeights(const std::vector<fl::Tensor<3>> &weights) override;

    int GetInputRank() const override;

    int GetOutputRank() const override;

private:
    // weight dims [heads, input_dim, dim]
    // for dev purposes, removed heads dim
    Tensor<3> w_q;
    Tensor<3> w_k;
    Tensor<3> w_v;

    Tensor<3> w_0;

    Eigen::Index heads = 1;
    Tensor<4> QK_T;
};

} // namespace fl

#include "multihead_attention.ipp"

#endif //FLARE_MULTIHEAD_ATTENTION_HPP
