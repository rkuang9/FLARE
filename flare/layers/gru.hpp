//
// Created by R on 2/8/23.
//

#ifndef FLARE_GRU_HPP
#define FLARE_GRU_HPP

#include "layer.hpp"
#include "rnn/gru_cell.hpp"

namespace fl
{

/**
 *
 * @tparam CandidateActivation       Output candidate activation function
 * @tparam GateActivation   Update and reset gates activation function
 * @tparam ReturnSequences  Return the entire time sequence or just the last one
 */
template<typename CandidateActivation = TanH,
        typename GateActivation = Sigmoid,
        bool ReturnSequences = true>
class GRU : public Layer
{
public:
    explicit GRU(int features_len, int outputs,
                 const Initializer<2> &initializer = GlorotUniform<2>());

    void Forward(const Tensor<3> &inputs) override;

    void Forward(const Layer &prev) override;

    void Backward(const Tensor<3> &gradients) override;

    void Backward(Layer &next) override;

    void Update(Optimizer &optimizer) override;

    const Tensor<2> &GetOutput2D() const override;

    const Tensor<3> &GetOutput3D() const override;

    const Tensor<2> &GetInputGradients2D() override;

    const Tensor<3> &GetInputGradients3D() override;

    // single matrix, horizontal order is update, reset, candidate,
    // with W on top, U on bottom
    void SetWeights(const std::vector<Tensor<2>> &weights) override;

    int GetInputRank() const override;

    int GetOutputRank() const override;

private:
    // weights w_zr are the update and reset gate weights stacked like
    // the following to save on computations:
    // W_z W_r
    // U_z U_r
    Tensor<2> w_zr;
    Tensor<2> w_c;
    Tensor<2> dL_dw_zr;
    Tensor<2> dL_dw_c;

    Tensor<3> x;
    Tensor<3> dL_dx;
    Tensor<3> h_candidate;
    Tensor<3> h;

    Eigen::Index input_len;
    Eigen::Index output_len;


    // each GRU cell perform their own forward,
    // backward weights, and backward input for their time step
    std::vector<GRUCell<CandidateActivation, GateActivation>> gru_cells;

    // multithreading
    Eigen::ThreadPool pool = Eigen::ThreadPool(
            (int) std::thread::hardware_concurrency());
    Eigen::ThreadPoolDevice device = Eigen::ThreadPoolDevice(&pool, 2);
};

} // namespace fl

#include "gru.ipp"

#endif //FLARE_GRU_HPP
