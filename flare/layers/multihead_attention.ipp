//
// Created by R on 4/23/23.
//

#include "multihead_attention.hpp"

namespace fl
{

MultiHeadAttention::MultiHeadAttention(
        Eigen::Index heads, Eigen::Index input_dim, Eigen::Index query_dim,
        Eigen::Index value_dim, const Initializer<3> &initializer) :
        Layer(3, 3),
        heads(heads),
        input_dim(input_dim),
        query_dim(query_dim)
{
    this->name = "multi_head_attention";

    this->w_q = initializer(Dims<3>(heads, input_dim, query_dim),
                            input_dim, query_dim);

    this->w_k = initializer(Dims<3>(heads, input_dim, query_dim),
                            input_dim, query_dim);

    this->w_v = initializer(Dims<3>(heads, input_dim, value_dim),
                            input_dim, value_dim);

    this->w_o = initializer(Dims<3>(heads, query_dim, input_dim),
                            query_dim, input_dim);
    this->w_o.setConstant(1.0); // set 1 for dev purposes
}


// Apply the formula Attention(Q,K,V) = softmax(QK_T/sqrt(d_k))V
// Dimension notation:
// N-batch, T-sequence/time, F-features, H-heads, D-dimensions(of query/key)
void MultiHeadAttention::Forward(const std::vector<fl::Tensor<3>> &inputs)
{
    FL_REQUIRES(inputs.size() == 3 &&
                inputs[0].dimension(0) == inputs[1].dimension(0) &&
                inputs[1].dimension(0) == inputs[2].dimension(0),
                this->name << " expected 3 inputs, all with the same batch size");

    ContractDim matmul {Axes(2, 1)};

    // [N,T,F] x [H,F,D] = [N,T,H,D]
    const Tensor<4> query = inputs[0].contract(this->w_q, matmul) /
                            std::sqrt(static_cast<Scalar>(this->query_dim));
    const Tensor<4> key = inputs[1].contract(this->w_k, matmul);
    const Tensor<4> value = inputs[2].contract(this->w_v, matmul);

    const Eigen::Index batches = query.dimension(0);
    const Eigen::Index sequence = query.dimension(1);
    const Eigen::Index dims = this->query_dim;

    // resize to [N,T,H,T]
    this->QK_T.resize(Dims<4>(batches, sequence, this->heads, sequence));
    this->QK_T.setZero();


    // Calculate query x key_transposed, [N,T,H,D] x [N,T,H,D] = [N,T,H,T]
    // contract along last dim while holding the batch,head dim constant
    // Since Eigen Tensor doesn't support Einstein summation notation, a vectorized
    // for-loop is (slightly) faster than slice and contracting
    // The einsum notation would be "abhc,adhc->abhd" (a=batch,h=head)
#ifdef _OPENMP
#pragma omp parallel for num_threads(2) default(none) shared(batches, sequence, key, query, dims)
#endif
    for (Eigen::Index batch = 0; batch < batches; batch++) {
        for (Eigen::Index row_l = 0; row_l < sequence; row_l++) {
            for (Eigen::Index row_r = 0; row_r < sequence; row_r++) {
                for (Eigen::Index head = 0; head < this->heads; head++) {
                    for (Eigen::Index col = 0; col < dims; col++) {
                        // dot product along the last dim (col)
                        QK_T(batch, row_l, head, row_r) +=
                                query(batch, row_l, head, col) *
                                key(batch, row_r, head, col);
                    }
                }
            }
        }
    }

    // apply softmax activation
    this->sm_QK_T.resize(this->QK_T.dimensions());
    this->sm_QK_T.device(this->device) = Softmax::Activate(this->QK_T);


    this->sm_QK_T_V.resize(value.dimensions());
    this->sm_QK_T_V.setZero();

    // Finish the attention calculation by multiplying with value
    // contracting [N,T,H,T] x [N,T,H,D] = [N,T,H,D]
#ifdef _OPENMP
#pragma omp parallel for num_threads(2) default(none) shared(batches, sequence, dims, value)
#endif
    for (Eigen::Index batch = 0; batch < batches; batch++) {
        for (Eigen::Index row_l = 0; row_l < sequence; row_l++) {
            for (Eigen::Index row_r = 0; row_r < sequence; row_r++) {
                for (Eigen::Index head = 0; head < this->heads; head++) {
                    for (Eigen::Index col = 0; col < dims; col++) {
                        // Dot product along the sm_QK_T's 3rd and value's 2nd dim
                        this->sm_QK_T_V(batch, row_l, head, col) +=
                                this->sm_QK_T(batch, row_l, head, row_r) *
                                value(batch, row_r, head, col);
                    }
                }
            }
        }
    }

    // concat all heads and multiply by output weights, this is equivalent to
    // the double contraction [N,T,H,D] x [H,D,F] = [N,T,F]
    Eigen::array<Eigen::IndexPair<int>, 2> double_contract = {
            Eigen::IndexPair<int>(2, 0), Eigen::IndexPair<int>(3, 1)};

    this->A.resize(inputs[0].dimensions());
    this->A.device(this->device) = sm_QK_T_V.contract(w_o, double_contract);

    /*
     // For-loop version of Forward() output before summing along heads dimension
#ifdef _OPENMP
#pragma omp parallel for num_threads(2) default(none) shared(batches, sequence, dims, pre_output, increment)
#endif
    for (Eigen::Index batch = 0; batch < batches; batch++) {
        for (Eigen::Index row = 0; row < sequence; row++) {
            for (Eigen::Index head = 0; head < this->heads; head++) {
                for (Eigen::Index col = 0; col < dims; col++) {
                    for (Eigen::Index f = 0; f < this->input_dim; f++) {
                        pre_output(batch, row, head, f) +=
                                this->sm_QK_T_V(batch, row, head, col) * w_o(head, col, f);
                    }
                }
            }
        }
    }
     */
}


void MultiHeadAttention::Forward(const Tensor<3> &inputs)
{

}


void MultiHeadAttention::Forward(const Layer &prev)
{

}


void MultiHeadAttention::Backward(const Tensor<2> &gradients)
{

}


void MultiHeadAttention::Backward(Layer &next)
{

}


void MultiHeadAttention::Update(Optimizer &optimizer)
{

}


const Tensor<3> &MultiHeadAttention::GetOutput3D() const
{
    return this->A;
}


const Tensor<3> &MultiHeadAttention::GetInputGradients3D()
{

}


std::vector<fl::Tensor<3>> MultiHeadAttention::GetWeights3D() const
{
    return {w_k, w_q, w_v, w_o};
}


std::vector<fl::Tensor<3>> MultiHeadAttention::GetWeightGradients3D() const
{
    return {dL_w_k, dL_w_q, dL_w_v, dL_w_o};
}


void MultiHeadAttention::SetWeights(const std::vector<fl::Tensor<3>> &weights)
{
    // Expects the order {w_query, w_key, w_value, w_output}
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

    if (weights[3].dimensions() != this->w_o.dimensions()) {
        error_msg << this->name
                  << " SetWeights() expected output weights dimensions "
                  << this->w_o.dimensions();
        throw std::invalid_argument(error_msg.str());
    }

    this->w_q.device(this->device) = weights[0];
    this->w_k.device(this->device) = weights[1];
    this->w_v.device(this->device) = weights[2];
    this->w_o.device(this->device) = weights[3];
}

} // namespace fl