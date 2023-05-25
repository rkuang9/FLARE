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

    // for functional programming, not to be used with Layer* and Sequential
    // inputs = [query, key, value]
    void Forward(const std::vector<fl::Tensor<3>> &inputs);

    // self attention where query = key = value
    void Forward(const Tensor<3> &inputs) override;

    void Forward(const Layer &prev) override;

    void Backward(const Tensor<2> &gradients) override;

    void Backward(Layer &next) override;

    void Update(Optimizer &optimizer) override;

    const Tensor<3> &GetOutput3D() const override;

    const Tensor<3> &GetInputGradients3D() override;

    std::vector<fl::Tensor<3>> GetWeights3D() const override;

    std::vector<fl::Tensor<3>> GetWeightGradients3D() const override;

    void SetWeights(const std::vector<fl::Tensor<3>> &weights) override;

private:
    // weight dims [heads, input_dim, dim]
    // for dev purposes, removed heads dim
    Tensor<3> w_q;
    Tensor<3> w_k;
    Tensor<3> w_v;
    Tensor<3> w_o;

    Tensor<3> dL_w_q;
    Tensor<3> dL_w_k;
    Tensor<3> dL_w_v;
    Tensor<3> dL_w_o;

    Eigen::Index heads = 1;
    Eigen::Index input_dim = -1;
    Eigen::Index query_dim = -1;

    Tensor<4> QK_T;
    Tensor<4> sm_QK_T;
    Tensor<4> sm_QK_T_V;
    Tensor<3> A; // softmax(Q x K_T / sqrt(dk)) x V

    Eigen::ThreadPoolDevice device = Eigen::ThreadPoolDevice(new Eigen::ThreadPool(
            (int) std::thread::hardware_concurrency()), 2);
};

} // namespace fl

#include "multihead_attention.ipp"

#endif //FLARE_MULTIHEAD_ATTENTION_HPP
