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
        query_dim(query_dim),
        key_value_dim(value_dim)
{
    this->name = "multi_head_attention";

    int fan_in = static_cast<int>(input_dim);
    int fan_out = static_cast<int>(query_dim);

    this->w_q = initializer(Dims<3>(heads, input_dim, query_dim), fan_in, fan_out);
    this->w_k = initializer(Dims<3>(heads, input_dim, query_dim), fan_in, fan_out);
    this->w_v = initializer(Dims<3>(heads, input_dim, value_dim), fan_in, fan_out);
    this->w_o = initializer(Dims<3>(heads, query_dim, input_dim), fan_in, fan_out);
}


// Apply the formula Attention(Q,K,V) = softmax(QK_T/sqrt(d_k))V
void MultiHeadAttention::Forward(const Tensor<3> &query,
                                 const Tensor<3> &key,
                                 const Tensor<3> &value)
{
    FL_REQUIRES(query.dimension(0) == key.dimension(0) &&
                key.dimension(0) == value.dimension(0),
                this->name << " expected 3 inputs, all with the same batch size");

    ContractDim matmul {Axes(2, 1)};

    // resize for multithreading with device to work
    this->Q.resize(query.dimension(0), query.dimension(1), this->heads,
                   this->query_dim);
    this->K.resize(key.dimension(0), key.dimension(1), this->heads,
                   this->key_value_dim);
    this->V.resize(value.dimension(0), value.dimension(1), this->heads,
                   this->key_value_dim);

    // [N,T,F] x [H,F,D] = [N,T,H,D]
    this->Q.device(this->device) = query.contract(this->w_q, matmul) /
                                   std::sqrt(static_cast<Scalar>(this->query_dim));
    this->K.device(this->device) = key.contract(this->w_k, matmul);
    this->V.device(this->device) = value.contract(this->w_v, matmul);

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
#pragma omp simd collapse(5)
#endif
    for (Eigen::Index batch = 0; batch < batches; batch++) {
        for (Eigen::Index row_l = 0; row_l < sequence; row_l++) {
            for (Eigen::Index row_r = 0; row_r < sequence; row_r++) {
                for (Eigen::Index head = 0; head < this->heads; head++) {
                    for (Eigen::Index col = 0; col < dims; col++) {
                        // dot product along the last dim (col)
                        QK_T(batch, row_l, head, row_r) +=
                                this->Q(batch, row_l, head, col) *
                                this->K(batch, row_r, head, col);
                    }
                }
            }
        }
    }

    // apply softmax activation
    this->sm_QK_T.resize(this->QK_T.dimensions());
    this->sm_QK_T.device(this->device) = Softmax::Activate(this->QK_T);


    this->sm_QK_T_V.resize(V.dimensions());
    this->sm_QK_T_V.setZero();

    // Finish the attention calculation by multiplying with value
    // contracting [N,T,H,T] x [N,T,H,D] = [N,T,H,D]
#ifdef _OPENMP
#pragma omp simd collapse(5)
#endif
    for (Eigen::Index batch = 0; batch < batches; batch++) {
        for (Eigen::Index seq_l = 0; seq_l < sequence; seq_l++) {
            for (Eigen::Index seq_r = 0; seq_r < sequence; seq_r++) {
                for (Eigen::Index head = 0; head < this->heads; head++) {
                    for (Eigen::Index col = 0; col < dims; col++) {
                        // Dot product along the sm_QK_T's 3rd and value's 2nd dim
                        this->sm_QK_T_V(batch, seq_l, head,
                                        col) += // TODO: seq_l might be the wrong one if query_dim != key_dim
                                this->sm_QK_T(batch, seq_l, head, seq_r) *
                                this->V(batch, seq_r, head, col);
                    }
                }
            }
        }
    }

    // concat all heads and multiply by output weights, this is equivalent to
    // the double contraction [N,T,H,D] x [H,D,F] = [N,T,F]
    Eigen::array<Eigen::IndexPair<int>, 2> double_contract = {
            Eigen::IndexPair<int>(2, 0), Eigen::IndexPair<int>(3, 1)};

    this->A.resize(query.dimensions());
    this->A.device(this->device) = sm_QK_T_V.contract(w_o, double_contract);
    /*
     // For-loop version of Forward() output before summing along heads dimension
#ifdef _OPENMP
#pragma omp simd collapse(5)
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
    this->Forward(inputs, inputs, inputs);
}


void MultiHeadAttention::Forward(const Layer &prev)
{
    this->Forward(prev.GetOutput3D(), prev.GetOutput3D(), prev.GetOutput3D());
}


// during the initial implementation, Eigen does not have einsum support
void MultiHeadAttention::Backward(const Tensor<3> &gradients)
{
    Eigen::array<Eigen::IndexPair<int>, 2> w_o_grad_contract = {
            Eigen::IndexPair<int>(0, 0), Eigen::IndexPair<int>(1, 1)};
    this->dL_w_o.device(this->device) =
            this->sm_QK_T_V.contract(gradients, w_o_grad_contract);
    Eigen::Index batch = gradients.dimension(0);
    Eigen::Index sequence = gradients.dimension(1);
    Eigen::Index seq_query = this->query_dim;
    Eigen::Index seq_key = this->key_value_dim;
    Eigen::Index seq_value = this->key_value_dim;
    Eigen::Index dims = this->query_dim;

    Tensor<4> dz_6(batch, sequence, heads, dims);
    dz_6.device(this->device) = gradients.contract(this->w_o,
                                                   ContractDim {Axes(2, 2)});


    Tensor<4> dz_3(batch, seq_query, this->heads, dims);
    dz_3.setZero();

#ifdef _OPENMP
#pragma omp simd collapse(5)
#endif
    for (Eigen::Index N = 0; N < batch; N++) {
        for (Eigen::Index Tk = 0; Tk < seq_key; Tk++) {
            for (Eigen::Index Tq = 0; Tq < seq_query; Tq++) {
                for (Eigen::Index H = 0; H < this->heads; H++) {
                    for (Eigen::Index D = 0; D < dims; D++) {
                        // einsum notation: NQHK,NQHD->NKHD
                        dz_3(N, Tk, H, D) +=
                                this->sm_QK_T(N, Tq, H, Tk) * dz_6(N, Tq, H, D);
                    }
                }
            }
        }
    }


    Tensor<4> dz_5(batch, seq_query, heads, seq_value);
    dz_5.setZero();

#ifdef _OPENMP
#pragma omp simd collapse(5)
#endif
    for (Eigen::Index N = 0; N < batch; N++) {
        for (Eigen::Index Tq = 0; Tq < seq_query; Tq++) {
            for (Eigen::Index H = 0; H < this->heads; H++) {
                for (Eigen::Index Tv = 0; Tv < seq_key; Tv++) {
                    for (Eigen::Index D = 0; D < dims; D++) {
                        // einsum notation: NQHV,NQHD->NVHD
                        dz_5(N, Tq, H, Tv) +=
                                dz_6(N, Tq, H, D) * this->V(N, Tv, H, D);
                    }
                }
            }
        }
    }

    Tensor<4> dz_4(batch, seq_query, this->heads, seq_key);
    dz_4.setZero();

    Tensor<5> softmax_grad(batch, seq_query, this->heads, seq_key, seq_key);
    softmax_grad.device(this->device) = Softmax::Gradients(this->sm_QK_T);

#ifdef _OPENMP
#pragma omp simd collapse(5)
#endif
    for (Eigen::Index N = 0; N < batch; N++) {
        for (Eigen::Index Tq = 0; Tq < seq_query; Tq++) {
            for (Eigen::Index H = 0; H < this->heads; H++) {
                for (Eigen::Index Tk = 0; Tk < seq_key; Tk++) {
                    for (Eigen::Index Tv = 0; Tv < dims; Tv++) {
                        // einsum notation: NQHV,NQHKK->NQHK
                        dz_4(N, Tq, H, Tk) += dz_5(N, Tq, H, Tv) *
                                              softmax_grad(N, Tq, H, Tk, Tk);
                    }
                }
            }
        }
    }


    Tensor<4> dz_2(batch, seq_key, this->heads, dims);
    dz_2.setZero();

#ifdef _OPENMP
#pragma omp simd collapse(5)
#endif
    for (Eigen::Index N = 0; N < batch; N++) {
        for (Eigen::Index Tq = 0; Tq < seq_query; Tq++) {
            for (Eigen::Index Tk = 0; Tk < seq_key; Tk++) {
                for (Eigen::Index H = 0; H < this->heads; H++) {
                    for (Eigen::Index D = 0; D < dims; D++) {
                        // einsum notation:
                        dz_2(N, Tk, H, D) += dz_4(N, Tq, H, Tk) *
                                             this->Q(N, Tq, H, D);
                    }
                }
            }
        }
    }


    Tensor<4> dz_1(batch, seq_query, this->heads, dims);
    dz_1.setZero();

#ifdef _OPENMP
#pragma omp simd collapse(5)
#endif
    for (Eigen::Index N = 0; N < batch; N++) {
        for (Eigen::Index Tq = 0; Tq < seq_query; Tq++) {
            for (Eigen::Index H = 0; H < this->heads; H++) {
                for (Eigen::Index D = 0; D < dims; D++) {
                    for (Eigen::Index Tk = 0; Tk < seq_key; Tk++) {
                        // einsum notation: NQHK,NKHD->NQHD
                        dz_1(N, Tq, H, D) += dz_4(N, Tq, H, Tk) *
                                             this->K(N, Tk, H, D);
                    }
                }
            }
        }
    }
}


void MultiHeadAttention::Backward(Layer &next)
{
    this->Backward(next.GetInputGradients3D());
}


void MultiHeadAttention::Update(Optimizer &optimizer)
{
    throw std::invalid_argument("Check MultiHeadAttention::Update");
    optimizer.Minimize(this->w_q, this->dL_w_q);
    optimizer.Minimize(this->w_k, this->dL_w_k);
    optimizer.Minimize(this->w_v, this->dL_w_v);
    optimizer.Minimize(this->w_o, this->dL_w_o);
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