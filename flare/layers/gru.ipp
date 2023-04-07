//
// Created by R on 2/8/23.
//

#include "gru.hpp"

namespace fl
{

template<typename Activation, typename GateActivation, bool ReturnSequences>
GRU<Activation, GateActivation, ReturnSequences>::GRU(
        int features_len, int outputs, const Initializer<2> &initializer)
        : output_len(outputs),
          input_len(features_len)
{
    this->name = "gru";

    this->w_zr = initializer.Initialize(
            Dims<2>(features_len + outputs, 2 * outputs), features_len, outputs);
    this->w_c = initializer.Initialize(
            Dims<2>(features_len + outputs, outputs), features_len, outputs);
    this->dL_dw_zr.resize(this->w_zr.dimensions());
    this->dL_dw_c.resize(this->w_c.dimensions());
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
void GRU<Activation, GateActivation, ReturnSequences>::Forward(
        const Tensor<3> &inputs)
{
    fl_assert(inputs.dimension(2) == this->input_len,
              this->name << " expected an input of "
                         << Dims<3>(-1, -1, this->input_len) << " got "
                         << inputs.dimensions() << " instead");

    this->x.resize(inputs.dimensions());
    this->x.device(this->device) = inputs;
    this->h.resize(inputs.dimension(0), inputs.dimension(1), this->output_len);

    // if needed, allocate more cells to match the input's time steps
    // it is possible to have more cells than the current mini-batch's time steps
    if (this->gru_cells.size() < inputs.dimension(1)) {
        int num_new_cells = inputs.dimension(1) - this->gru_cells.size();

        for (int i = 0; i < num_new_cells; i++) {
            this->gru_cells.push_back(GRUCell<Activation, GateActivation>(
                    this->gru_cells.size(), inputs.dimension(2), this->output_len));
        }
    }

    for (int i = 0; i < inputs.dimension(1); i++) {
        this->gru_cells[i].Forward(inputs, this->h, this->w_zr,
                                   this->w_c, this->device);
    }

    if constexpr (!ReturnSequences) {
        this->h_no_seq.resize(inputs.dimension(0), this->output_len);
        this->h_no_seq.device(device) = this->h.chip(inputs.dimension(1) - 1, 1);
    }
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
void
GRU<Activation, GateActivation, ReturnSequences>::Forward(const Layer &prev)
{
    this->Forward(prev.GetOutput3D());
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
void GRU<Activation, GateActivation, ReturnSequences>::Backward(
        const Tensor<ReturnSequences ? 3 : 2> &gradients)
{
    int time_steps = static_cast<int>(this->x.dimension(1));

    this->dL_dw_zr.setZero();
    this->dL_dw_c.setZero();

    if constexpr(ReturnSequences) {
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
    else {
        this->gru_cells[time_steps - 1].Backward(
                gradients.reshape(
                        Dims<3>(this->x.dimension(0), 1, this->output_len)),
                this->x, this->w_zr, this->dL_dw_zr,
                this->w_c, this->dL_dw_c,
                this->h, this->device);

        for (int i = time_steps - 2; i >= 0; --i) {
            this->gru_cells[i].Backward(
                    this->gru_cells[i + 1].GetCellInputGradients(),
                    this->x, this->w_zr, this->dL_dw_zr,
                    this->w_c, this->dL_dw_c,
                    this->h, this->device);
        }
    }
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
void GRU<Activation, GateActivation, ReturnSequences>::Backward(Layer &next)
{
    this->Backward(next.GetInputGradients3D());
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
void GRU<Activation, GateActivation, ReturnSequences>::Update(
        Optimizer &optimizer)
{
    optimizer.Minimize(this->w_zr, this->dL_dw_zr);
    optimizer.Minimize(this->w_c, this->dL_dw_c);
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
const Tensor<2> &
GRU<Activation, GateActivation, ReturnSequences>::GetOutput2D() const
{
    if constexpr(ReturnSequences) {
        throw std::logic_error(
                this->name +
                " was initialized with ReturnSequences=true, use GetOutput3D instead");
    }

    return this->h_no_seq;
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
const Tensor<3> &
GRU<Activation, GateActivation, ReturnSequences>::GetOutput3D() const
{
    if constexpr(!ReturnSequences) {
        throw std::logic_error(
                this->name +
                " was initialized with ReturnSequences=false, use GetOutput2D instead");
    }

    return this->h;
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
const Tensor<3> &
GRU<Activation, GateActivation, ReturnSequences>::GetInputGradients3D()
{
    this->dL_dx.resize(this->x.dimensions());
    int time_steps = static_cast<int>(this->x.dimension(1));

    for (int i = time_steps - 1; i >= 0; --i) {
        this->gru_cells[i].CalcLayerInputGradients(this->w_zr, this->w_c,
                                                   this->dL_dx, this->device);
    }

    return this->dL_dx;
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
std::vector<fl::Tensor<2>>
GRU<Activation, GateActivation, ReturnSequences>::GetWeights2D() const
{
    return {this->w_zr.concatenate(this->w_c, 1)};
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
std::vector<fl::Tensor<2>>
GRU<Activation, GateActivation, ReturnSequences>::GetWeightGradients2D() const
{
    return {this->dL_dw_zr.concatenate(this->dL_dw_c, 1)};
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
void GRU<Activation, GateActivation, ReturnSequences>::SetWeights(
        const std::vector<fl::Tensor<2>> &weights)
{
    if (weights.front().dimension(0) != (this->input_len + this->output_len) ||
        weights.front().dimension(1) != this->output_len * 3) {
        throw std::invalid_argument(
                this->name + " SetWeights() expects the tensor shape [" +
                std::to_string(this->input_len + this->output_len) + ", " +
                std::to_string(this->output_len * 3) + "]");
    }

    Dims<2> w_zr_offset(0, 0);
    Dims<2> w_zr_extent(this->input_len + this->output_len, this->output_len * 2);

    Dims<2> w_c_offset(0, this->output_len * 2);
    Dims<2> w_c_extent(this->input_len + this->output_len, this->output_len);

    this->w_zr = weights.front().slice(w_zr_offset, w_zr_extent);
    this->w_c = weights.front().slice(w_c_offset, w_c_extent);
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
int GRU<Activation, GateActivation, ReturnSequences>::GetInputRank() const
{
    return 0;
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
int GRU<Activation, GateActivation, ReturnSequences>::GetOutputRank() const
{
    return 0;
}


} // namespace fl