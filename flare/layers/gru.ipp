//
// Created by R on 2/8/23.
//

#include "gru.hpp"

namespace fl
{

template<typename CandidateActivation, typename GateActivation, bool ReturnSequences>
GRU<CandidateActivation, GateActivation, ReturnSequences>::GRU(
        int features_len, int outputs, const Initializer<2> &initializer)
        : output_len(outputs),
          input_len(features_len)
{
    if (!ReturnSequences) {
        throw std::invalid_argument(
                "GRU ReturnSequences=false is not supported yet");
    }

    this->name = "gru";

    this->w_zr = initializer.Initialize(
            Dims<2>(features_len + outputs, 2 * outputs), features_len, outputs);
    this->w_c = initializer.Initialize(
            Dims<2>(features_len + outputs, outputs), features_len, outputs);
    this->dL_dw_zr.resize(this->w_zr.dimensions());
    this->dL_dw_c.resize(this->w_c.dimensions());
    w_zr.setConstant(1.0);
    w_c.setConstant(1.0);
}


template<typename CandidateActivation, typename GateActivation, bool ReturnSequences>
void GRU<CandidateActivation, GateActivation, ReturnSequences>::Forward(
        const Tensor<3> &inputs)
{
    fl_assert(inputs.dimension(2) == this->input_len,
              this->name << " expected an input of "
                            << Dims<3>(-1, -1, this->input_len) << " got "
                            << inputs.dimensions() << " instead");
    // layer output dimensions are [batch, time, output] but
    // are initially sized as [batch, time+1, output]
    // where the extra +1 is the initial h_0 tensor of zeroes
    this->x = inputs;
    this->h.resize(inputs.dimension(0), inputs.dimension(1), this->output_len);

    // if needed, allocate more cells to match the input's time steps
    // it is possible to have more cells than the current mini-batch's time steps
    if (this->gru_cells.size() < inputs.dimension(1)) {
        int num_new_cells = inputs.dimension(1) - this->gru_cells.size();

        for (int i = 0; i < num_new_cells; i++) {
            this->gru_cells.push_back(GRUCell<CandidateActivation, GateActivation>(
                    this->gru_cells.size(), inputs.dimension(0),
                    inputs.dimension(2), this->output_len));
        }
    }

    for (int i = 0; i < inputs.dimension(1); i++) {
        this->gru_cells[i].Forward(inputs, this->h, this->w_zr,
                                   this->w_c, this->device);
    }
}


template<typename CandidateActivation, typename GateActivation, bool ReturnSequences>
void
GRU<CandidateActivation, GateActivation, ReturnSequences>::Forward(const Layer &prev)
{
    this->Forward(prev.GetOutput3D());
}


template<typename CandidateActivation, typename GateActivation, bool ReturnSequences>
void GRU<CandidateActivation, GateActivation, ReturnSequences>::Backward(
        const Tensor<3> &gradients)
{
    int time_steps = static_cast<int>(this->x.dimension(1));

    this->gru_cells[time_steps - 1].Backward(
            gradients.slice(Dims<3>(0, time_steps - 1, 0),
                            Dims<3>(this->x.dimension(0), 1, this->output_len)),
            this->x, this->w_zr, this->dL_dw_zr,
            this->w_c, this->dL_dw_c,
            this->h, this->device);

    for (int i = time_steps - 2; i >= 0; --i) {
        auto grads = gradients.slice(
                Dims<3>(0, i, 0),
                Dims<3>(this->x.dimension(0), 1, this->output_len)) +
                     this->gru_cells[i + 1].GetCellInputGradients();

        this->gru_cells[i].Backward(grads, this->x, this->w_zr, this->dL_dw_zr,
                                    this->w_c, this->dL_dw_c,
                                    this->h, this->device);
    }
}


template<typename CandidateActivation, typename GateActivation, bool ReturnSequences>
void GRU<CandidateActivation, GateActivation, ReturnSequences>::Backward(Layer &next)
{
    this->Backward(next.GetInputGradients3D());
}


template<typename CandidateActivation, typename GateActivation, bool ReturnSequences>
void GRU<CandidateActivation, GateActivation, ReturnSequences>::Update(
        Optimizer &optimizer)
{
    optimizer.Minimize(this->w_zr, this->dL_dw_zr);
    optimizer.Minimize(this->w_c, this->dL_dw_c);

    this->dL_dw_zr.setZero();
    this->dL_dw_c.setZero();
}


template<typename CandidateActivation, typename GateActivation, bool ReturnSequences>
const Tensor<2> &
GRU<CandidateActivation, GateActivation, ReturnSequences>::GetOutput2D() const
{
    throw std::invalid_argument("not implemented yet");
    return this->h;
}


template<typename CandidateActivation, typename GateActivation, bool ReturnSequences>
const Tensor<3> &
GRU<CandidateActivation, GateActivation, ReturnSequences>::GetOutput3D() const
{
    return this->h;
}


template<typename CandidateActivation, typename GateActivation, bool ReturnSequences>
const Tensor<2> &
GRU<CandidateActivation, GateActivation, ReturnSequences>::GetInputGradients2D()
{
    return Layer::GetInputGradients2D();
}


template<typename CandidateActivation, typename GateActivation, bool ReturnSequences>
const Tensor<3> &
GRU<CandidateActivation, GateActivation, ReturnSequences>::GetInputGradients3D()
{
    this->dL_dx.resize(this->x.dimensions());
    int time_steps = static_cast<int>(this->x.dimension(1));

    for (int i = time_steps - 1; i >= 0; --i) {
        this->gru_cells[i].CalcLayerInputGradients(this->w_zr, this->w_c,
                                                   this->dL_dx, this->device);
    }

    return this->dL_dx;
}


template<typename CandidateActivation, typename GateActivation, bool ReturnSequences>
void GRU<CandidateActivation, GateActivation, ReturnSequences>::SetWeights(
        const Tensor<2> &weights)
{
    if (weights.dimension(0) != (this->input_len + this->output_len) ||
        weights.dimension(1) != this->output_len * 3) {
        throw std::invalid_argument(
                this->name + " expects the tensor shape [" +
                std::to_string(this->input_len + this->output_len) + ", " +
                std::to_string(this->output_len * 3) + "]");
    }

    Dims<2> w_zr_offset(0, 0);
    Dims<2> w_c_offset(0, this->output_len * 2);

    Dims<2> w_zr_extent(this->input_len + this->output_len, this->output_len * 2);
    Dims<2> w_c_extent(this->input_len + this->output_len, this->output_len);

    this->w_zr = weights.slice(w_zr_offset, w_zr_extent);
    this->w_c = weights.slice(w_c_offset, w_c_extent);
}


template<typename CandidateActivation, typename GateActivation, bool ReturnSequences>
int GRU<CandidateActivation, GateActivation, ReturnSequences>::GetInputRank() const
{
    return 3;
}


template<typename CandidateActivation, typename GateActivation, bool ReturnSequences>
int GRU<CandidateActivation, GateActivation, ReturnSequences>::GetOutputRank() const
{
    if constexpr (ReturnSequences) {
        return 3;
    }
    else {
        return 2;
    }
}


} // namespace fl