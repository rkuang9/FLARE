//
// Created by R on 4/23/23.
//

#ifndef FLARE_MULTIHEAD_ATTENTION_HPP
#define FLARE_MULTIHEAD_ATTENTION_HPP

#include "layer.hpp"

namespace fl
{

// Dimensions notation for documentation
// N - batch
// T - sequence
// H - number of heads
// F - input features (typically the last dimension of inputs)
// D - dimension of query or key
class MultiHeadAttention : public Layer
{
public:
    /**
     *
     * For now dim is the size of query/key/value's last dimension
     *
     * @param input_dim   number of features in input
     * @param key_dim     size of query and key dimensions per head
     * @param value_dim   size of value dimension per head
     * @param initializer
     */
    MultiHeadAttention(Eigen::Index heads, Eigen::Index input_dim, Eigen::Index dim,
                       const Initializer<3> &initializer = GlorotUniform<3>());

    // not to be used with Layer* and Sequential
     void Forward(const Tensor<3> &query,
                 const Tensor<3> &key,
                 const Tensor<3> &value);

    // self attention where query = key = value
    void Forward(const Tensor<3> &inputs) override;

    void Forward(const Layer &prev) override;

    void Backward(const Tensor<3> &gradients) override;

    void Backward(Layer &next) override;

    void Update(Optimizer &optimizer) override;

    const Tensor<3> &GetOutput3D() const override;

    const Tensor<3> &GetInputGradients3D() override;

    std::vector<fl::Tensor<3>> GetWeights3D() const override;

    std::vector<fl::Tensor<3>> GetWeightGradients3D() const override;

    void SetWeights(const std::vector<fl::Tensor<3>> &weights) override;

private:
    // layer input, not sure if having these as pointers is safe,
    // but it's faster keeping copies per Forward() and requires less memory
    Tensor<3> const *q = nullptr;
    Tensor<3> const *k = nullptr;
    Tensor<3> const *v = nullptr;

    // layer input multiplied with weights
    Tensor<4> Q;
    Tensor<4> K;
    Tensor<4> V;

    // weight dims [heads, input_dim, dim]
    Tensor<3> w_q;
    Tensor<3> w_k;
    Tensor<3> w_v;
    Tensor<3> w_o;

    Tensor<3> dL_w_q;
    Tensor<3> dL_w_k;
    Tensor<3> dL_w_v;
    Tensor<3> dL_w_o;

    Eigen::Index heads = 1;
    Eigen::Index input_dim = -1; // length of input's last dimension
    Eigen::Index layer_dim = -1; // length of Q,K's last dimension

    // intermediate terms saved during Forward() for Backward()
    Tensor<4> sm_QK_T;
    Tensor<4> sm_QK_T_V;
    Tensor<3> A; // softmax(Q x K_T / sqrt(dk)) x V

    Eigen::ThreadPool pool {(int) std::thread::hardware_concurrency()};
    Eigen::ThreadPoolDevice device {&pool, 2};
};

} // namespace fl

#include "multihead_attention.ipp"

#endif //FLARE_MULTIHEAD_ATTENTION_HPP
