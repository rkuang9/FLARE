//
// Created by R on 4/23/23.
//

#include "multihead_attention.hpp"

namespace fl
{

MultiHeadAttention::MultiHeadAttention(
        Eigen::Index heads, Eigen::Index input_dim, Eigen::Index layer_dim,
        const Initializer<3> &initializer) :
        heads(heads),
        input_dim(input_dim),
        layer_dim(layer_dim)
{
    this->name = "multihead_attention";

    int fan_in = static_cast<int>(input_dim);
    int fan_out = static_cast<int>(layer_dim);

    this->w_q = initializer(Dims<3>(heads, input_dim, layer_dim), fan_in, fan_out);
    this->w_k = initializer(Dims<3>(heads, input_dim, layer_dim), fan_in, fan_out);
    this->w_v = initializer(Dims<3>(heads, input_dim, layer_dim), fan_in, fan_out);

    this->w_o = initializer(Dims<3>(heads, layer_dim, input_dim), fan_in, fan_out);

    this->dL_w_q.resize(this->w_q.dimensions());
    this->dL_w_k.resize(this->w_k.dimensions());
    this->dL_w_v.resize(this->w_v.dimensions());
    this->dL_w_o.resize(this->w_o.dimensions());
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

    this->q = &query;
    this->k = &key;
    this->v = &value;

    // resize for multithreading with device to work
    this->Q.resize(query.dimension(0), query.dimension(1), this->heads,
                   this->layer_dim);
    this->K.resize(key.dimension(0), key.dimension(1), this->heads,
                   this->layer_dim);
    this->V.resize(value.dimension(0), value.dimension(1), this->heads,
                   this->layer_dim);

    // [N,T,F] x [H,F,D] = [N,T,H,D]
    this->Q.device(this->device) = query.contract(this->w_q, matmul);
    this->Q.device(this->device) =
            this->Q / std::sqrt(static_cast<Scalar>(this->layer_dim));
    this->K.device(this->device) = key.contract(this->w_k, matmul);
    this->V.device(this->device) = value.contract(this->w_v, matmul);

    const Eigen::Index batches = query.dimension(0);
    const Eigen::Index sequence = query.dimension(1);
    const Eigen::Index dims = this->layer_dim;

    // calculate QK^T, then apply softmax
    this->sm_QK_T.resize(Dims<4>(batches, sequence, this->heads, sequence));
    this->sm_QK_T.setZero();

#pragma omp parallel for simd shared(batches, sequence, dims) default(none) schedule(guided)
    for (Eigen::Index N = 0; N < batches; N++) {
        for (Eigen::Index Tq = 0; Tq < sequence; Tq++) {
            for (Eigen::Index H = 0; H < this->heads; H++) {
                for (Eigen::Index D = 0; D < dims; D++) {
                    for (Eigen::Index Tk = 0; Tk < sequence; Tk++) {
                        // einsum NQHK,NQHD->NKHD
                        this->sm_QK_T(N, Tq, H, Tk) +=
                                this->Q(N, Tq, H, D) *
                                this->K(N, Tk, H, D);
                    }
                }
            }
        }
    }

    this->sm_QK_T.device(this->device) = Softmax::Activate(this->sm_QK_T);

    this->sm_QK_T_V.resize(this->V.dimensions());
    this->sm_QK_T_V.setZero();

#pragma omp parallel for simd shared(batches, sequence, dims) default(none) schedule(guided)
    for (Eigen::Index N = 0; N < batches; N++) {
        for (Eigen::Index Tq = 0; Tq < sequence; Tq++) {
            for (Eigen::Index Tv = 0; Tv < sequence; Tv++) {
                for (Eigen::Index H = 0; H < this->heads; H++) {
                    for (Eigen::Index D = 0; D < dims; D++) {
                        // einsum NQHV,NVHD->NQHD
                        this->sm_QK_T_V(N, Tq, H, D) +=
                                this->sm_QK_T(N, Tq, H, Tv) * // here Tv is Tk
                                this->V(N, Tv, H, D);
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
}


void MultiHeadAttention::Forward(const Tensor<3> &inputs)
{
    this->Forward(inputs, inputs, inputs);
}


void MultiHeadAttention::Forward(const Layer &prev)
{
    this->Forward(prev.GetOutput3D(), prev.GetOutput3D(), prev.GetOutput3D());
}


// during this initial implementation, Eigen didn't have einsum
void MultiHeadAttention::Backward(const Tensor<3> &gradients)
{
    // TODO: FOR ALL INTERMEDIATE BACKPROPAGATION (dz_x) TERMS, MOVE
    // TODO: THEM TO CLASS MEMBERS TO AVOID REALLOCATION PER FUNCTION CALL
    this->dL_w_q.setZero();
    this->dL_w_k.setZero();
    this->dL_w_v.setZero();
    this->dL_w_o.setZero();

    const Eigen::Index batch = gradients.dimension(0);
    const Eigen::Index sequence = gradients.dimension(1);
    const Eigen::Index seq_query = this->Q.dimension(1);
    const Eigen::Index seq_key = this->K.dimension(1);
    const Eigen::Index seq_value = this->V.dimension(1);
    const Eigen::Index dims = this->layer_dim;


    Tensor<4> dz6(batch, sequence, this->heads, dims);
    dz6.device(this->device) = gradients.contract(this->w_o,
                                                  ContractDim {Axes(2, 2)});

    this->dL_dV.resize(batch, this->heads, dims, seq_key);
    this->dL_dV.setZero();

#pragma omp parallel for simd collapse(2) shared(batch, seq_query, dims, seq_key, dz6) default(none)
    for (Eigen::Index N = 0; N < batch; N++) {
        for (Eigen::Index Tq = 0; Tq < seq_query; Tq++) {
            for (Eigen::Index H = 0; H < this->heads; H++) {
                for (Eigen::Index D = 0; D < dims; D++) {
                    for (Eigen::Index Tk = 0; Tk < seq_key; Tk++) {
                        // moved Tk dim to the end in [N, Tk, H, D] for better cache access
                        this->dL_dV(N, H, D, Tk) +=
                                this->sm_QK_T(N, Tq, H, Tk) * dz6(N, Tq, H, D);
                    }
                }
            }
        }
    }

    Tensor<4> dz5(batch, seq_query, this->heads, seq_value);
    dz5.setZero();

#pragma omp parallel for simd shared(batch, seq_key, seq_query, dims, dz6, dz5) default(none)
    for (Eigen::Index N = 0; N < batch; N++) {
        for (Eigen::Index Tq = 0; Tq < seq_query; Tq++) {
            for (Eigen::Index H = 0; H < this->heads; H++) {
                for (Eigen::Index Tv = 0; Tv < seq_key; Tv++) {
                    for (Eigen::Index D = 0; D < dims; D++) {
                        dz5(N, Tq, H, Tv) +=
                                dz6(N, Tq, H, D) * this->V(N, Tv, H, D);
                    }
                }
            }
        }
    }
    // TODO: dz5 CAN BE COMBINED INTO WITH dz4 LOOP
    Tensor<4> dz4(batch, seq_key, this->heads, seq_key);
    dz4.setZero();

    Tensor<5> softmax_grad(batch, seq_query, this->heads, seq_key, seq_value);
    softmax_grad.device(this->device) = Softmax::Gradients(this->sm_QK_T);

#pragma omp parallel for simd collapse(3) shared(batch, seq_key, seq_query, seq_value, dims, dz5, dz4, softmax_grad) default(none)
    for (Eigen::Index N = 0; N < batch; N++) {
        for (Eigen::Index Tq = 0; Tq < seq_query; Tq++) {
            for (Eigen::Index H = 0; H < this->heads; H++) {
                for (Eigen::Index Tk = 0; Tk < seq_key; Tk++) {
                    for (Eigen::Index Tv = 0; Tv < seq_value; Tv++) {
                        dz4(N, Tq, H, Tk) +=
                                dz5(N, Tq, H, Tv) *
                                softmax_grad(N, Tq, H, Tk, Tv);
                    }
                }
            }
        }
    }

    // transpose dz4 for better cache access
    Tensor<4> dz4_transpose(dz4.dimension(0),
                            dz4.dimension(1),
                            dz4.dimension(3),
                            dz4.dimension(2));
    dz4_transpose.device(this->device) = dz4.shuffle(Dims<4>(0, 1, 3, 2));

    this->dL_dK.resize(batch, seq_key, this->heads, dims);
    this->dL_dK.setZero();

#pragma omp parallel for simd collapse(2) shared(batch, seq_key, seq_query, dims, dz4_transpose) default(none)
    for (Eigen::Index N = 0; N < batch; N++) {
        for (Eigen::Index Tq = 0; Tq < seq_query; Tq++) {
            for (Eigen::Index Tk = 0; Tk < seq_key; Tk++) {
                for (Eigen::Index H = 0; H < this->heads; H++) {
                    for (Eigen::Index D = 0; D < dims; D++) {
                        this->dL_dK(N, Tk, H, D) += dz4_transpose(N, Tq, Tk, H) *
                                                    this->Q(N, Tq, H, D);
                    }
                }
            }
        }
    }

    this->dL_dQ.resize(batch, seq_query, this->heads, dims);
    this->dL_dQ.setZero();

#pragma omp parallel for simd shared(batch, seq_key, seq_query, dims, dz4_transpose) default(none)
    for (Eigen::Index N = 0; N < batch; N++) {
        for (Eigen::Index Tq = 0; Tq < seq_query; Tq++) {
            for (Eigen::Index Tk = 0; Tk < seq_key; Tk++) {
                for (Eigen::Index H = 0; H < this->heads; H++) {
                    for (Eigen::Index D = 0; D < dims; D++) {
                        this->dL_dQ(N, Tq, H, D) +=
                                dz4_transpose(N, Tq, Tk, H) * this->K(N, Tk, H, D);
                    }
                }
            }
        }
    }

    // calculate weight gradients
    Eigen::array<Eigen::IndexPair<int>, 2> double_contract = {
            Eigen::IndexPair<int>(0, 0), Eigen::IndexPair<int>(1, 1)};
    Dims<3> shuffle_FHD_HFD(1, 0, 2);

    this->dL_w_q.device(this->device) =
            this->q->contract(this->dL_dQ, double_contract).shuffle(shuffle_FHD_HFD)
            / std::sqrt(static_cast<Scalar>(this->layer_dim));

    this->dL_w_k.device(this->device) =
            this->k->contract(this->dL_dK, double_contract).shuffle(shuffle_FHD_HFD);

    // dL_dV was "shuffled" so this will not be using the same contraction dims
    this->dL_w_v.device(this->device) =
            this->v->contract(this->dL_dV, Eigen::array<Eigen::IndexPair<int>, 2> {
                    Eigen::IndexPair<int>(0, 0),
                    Eigen::IndexPair<int>(1, 3)}).shuffle(shuffle_FHD_HFD);

    this->dL_w_o.device(this->device) =
            this->sm_QK_T_V.contract(gradients, double_contract);
}


void MultiHeadAttention::Backward(Layer &next)
{
    this->Backward(next.GetInputGradients3D());
}


void MultiHeadAttention::Update(Optimizer &optimizer)
{
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
    FL_REQUIRES(this->q == this->k && this->k == this->v, this->name
            << "::GetInputGradients3D should only be used for self-attention, when query, key, and value are the same");

    Eigen::array<Eigen::IndexPair<int>, 2> contract_axes_q_k = {
            Eigen::IndexPair<int>(2, 0), Eigen::IndexPair<int>(3, 2)};

    // using different contraction axes for value because dL_dV was
    // transposed during Backward() for better performance
    Eigen::array<Eigen::IndexPair<int>, 2> contract_axes_v = {
            Eigen::IndexPair<int>(1, 0), Eigen::IndexPair<int>(2, 2)};

    this->dL_dX.resize(this->q->dimensions());
    this->dL_dX.device(this->device) =
            this->dL_dQ.contract(this->w_q, contract_axes_q_k) /
            std::sqrt(static_cast<Scalar>(this->layer_dim)) +
            this->dL_dK.contract(this->w_k, contract_axes_q_k) +
            this->dL_dV.contract(this->w_v, contract_axes_v);

    return this->dL_dX;
}


std::vector<fl::Tensor<3> *> MultiHeadAttention::GetInputGradients()
{
    Eigen::array<Eigen::IndexPair<int>, 2> contract_axes_q_k = {
            Eigen::IndexPair<int>(2, 0), Eigen::IndexPair<int>(3, 2)};

    // using different contraction axes for value because dL_dV was
    // transposed during Backward() for better performance
    Eigen::array<Eigen::IndexPair<int>, 2> contract_axes_v = {
            Eigen::IndexPair<int>(1, 0), Eigen::IndexPair<int>(2, 2)};

    this->dL_dq.device(this->device) =
            this->dL_dQ.contract(this->w_q, contract_axes_q_k) /
            std::sqrt(static_cast<Scalar>(this->layer_dim));
    this->dL_dk.device(this->device) =
            this->dL_dK.contract(this->w_k, contract_axes_q_k);
    this->dL_dv.device(this->device) =
            this->dL_dV.contract(this->w_v, contract_axes_v);

    return {&this->dL_dq, &this->dL_dk, &this->dL_dv};
}


std::vector<fl::Tensor<3>> MultiHeadAttention::GetWeights3D() const
{
    return {w_q, w_k, w_v, w_o};
}


std::vector<fl::Tensor<3>> MultiHeadAttention::GetWeightGradients3D() const
{
    return {dL_w_q, dL_w_k, dL_w_v, dL_w_o};
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

    this->w_q = weights[0];
    this->w_k = weights[1];
    this->w_v = weights[2];
    this->w_o = weights[3];
}

} // namespace fl