//
// Created by R on 4/23/23.
//

#include "multihead_attention.hpp"

namespace fl
{

MultiHeadAttention::MultiHeadAttention(
        Eigen::Index heads, Eigen::Index input_dim, Eigen::Index query_dim,
        Eigen::Index value_dim, const Initializer<3> &initializer)
{
    this->name = "multi_head_attention";

    this->w_q = initializer(Dims<3>(heads, input_dim, query_dim),
                            input_dim, query_dim);

    this->w_k = initializer(Dims<3>(heads, input_dim, query_dim),
                            input_dim, query_dim);

    this->w_v = initializer(Dims<3>(heads, input_dim, value_dim),
                            input_dim, value_dim);

    this->w_0 = initializer(Dims<3>(heads, query_dim, input_dim),
                            query_dim, input_dim);
    this->w_0.setConstant(1.0); // set 1 for dev purposes
}


// Apply the formula Attention(Q,K,V) = softmax(QK_T/sqrt(d_k))V
// For now sqrt(d_k) is ignored until we figure out what it is
void MultiHeadAttention::Forward(const std::vector<fl::Tensor<3>> &inputs)
{
    FL_REQUIRES(inputs.size() == 3 &&
                inputs[0].dimension(0) == inputs[1].dimension(1) &&
                inputs[1].dimension(0) == inputs[2].dimension(0),
                this->name << " expected 3 inputs, all with the same batch size");

    ContractDim matmul {Axes(2, 1)};
    std::cout << this->name << "::Forward\n";
    const Tensor<4> query = inputs[0].contract(this->w_q, matmul);
    const Tensor<4> key = inputs[1].contract(this->w_k, matmul);
    const Tensor<4> value = inputs[2].contract(this->w_v, matmul);

    std::cout << "query: " << query.dimensions() << "\n" << query << "\n";
    std::cout << "key: " << key.dimensions() << "\n" << key << "\n";


    // resize to [batch, sequence, heads, sequence]
    this->QK_T.resize(Dims<4>(query.dimension(0),
                              query.dimension(1),
                              this->heads,
                              query.dimension(1)));
    this->QK_T.setZero();


    // since Eigen Tensor doesn't have an equivalent of np.einsum/tf.einsum,
    // a for-loop is faster than slice and contracting over batch and heads
    // the einsum notation would be "abhc,adhc->abhd" (a=batch,h=head)
#ifdef _OPENMP
    //#pragma omp parallel for num_threads(2) default(none) shared(heads, qk_t, device, query1, key1, matmul)
#endif
    for (Eigen::Index batch = 0; batch < query.dimension(0); batch++) {
        for (Eigen::Index row_l = 0; row_l < query.dimension(1); row_l++) {
            for (Eigen::Index row_r = 0; row_r < query.dimension(1); row_r++) {
                for (Eigen::Index head = 0; head < query.dimension(2); head++) {
                    for (Eigen::Index col = 0; col < query.dimension(3); col++) {
                        QK_T(batch, row_l, head, row_r) +=
                                query(batch, row_l, head, col) *
                                key(batch, row_r, head, col);
                    }
                }
            }
        }
    }

    std::cout << "QK_T\n" << QK_T << "\n";
}


void MultiHeadAttention::Forward(const Tensor<3> &inputs)
{

}


void MultiHeadAttention::Forward(const Layer &prev)
{
    Layer::Forward(prev);
}


void MultiHeadAttention::Backward(const Tensor<2> &gradients)
{
    Layer::Backward(gradients);
}


void MultiHeadAttention::Backward(Layer &next)
{
    Layer::Backward(next);
}


void MultiHeadAttention::Update(Optimizer &optimizer)
{

}


const Tensor<2> &MultiHeadAttention::GetOutput2D() const
{
    return Layer::GetOutput2D();
}


const Tensor<2> &MultiHeadAttention::GetInputGradients2D()
{
    return Layer::GetInputGradients2D();
}


std::vector<fl::Tensor<2>> MultiHeadAttention::GetWeights2D() const
{
    return Layer::GetWeights2D();
}


std::vector<fl::Tensor<2>> MultiHeadAttention::GetWeightGradients2D() const
{
    return Layer::GetWeightGradients2D();
}


void MultiHeadAttention::SetWeights(const std::vector<fl::Tensor<3>> &weights)
{
    // expects weights in the order: query, key, value,
    if (weights.size() != 4) {
        throw std::invalid_argument(
                this->name +
                " SetWeights() expects 4 tensors: query, key, value, and output weights");
    }

    std::ostringstream error_msg;

    if (weights[0].dimensions() != this->w_q.dimensions()) {
        error_msg << this->name << " SetWeights() expected query weights dimensions "
                  << this->w_q.dimensions();
        throw std::invalid_argument(error_msg.str());
    }

    if (weights[1].dimensions() != this->w_k.dimensions()) {
        error_msg << this->name << " SetWeights() expected key weights dimensions "
                  << this->w_k.dimensions();
        throw std::invalid_argument(error_msg.str());
    }

    if (weights[2].dimensions() != this->w_v.dimensions()) {
        error_msg << this->name << " SetWeights() expected value weights dimensions "
                  << this->w_v.dimensions();
        throw std::invalid_argument(error_msg.str());
    }

    if (weights[3].dimensions() != this->w_0.dimensions()) {
        error_msg << this->name
                  << " SetWeights() expected output weights dimensions "
                  << this->w_0.dimensions();
        throw std::invalid_argument(error_msg.str());
    }

    this->w_q = weights[0];
    this->w_k = weights[1];
    this->w_v = weights[2];
    this->w_0 = weights[3];
}


int MultiHeadAttention::GetInputRank() const
{
    return 3;
}


int MultiHeadAttention::GetOutputRank() const
{
    return 3;
}

} // namespace fl