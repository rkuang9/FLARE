//
// Created by R on 3/25/23.
//

#include "lstm.hpp"

namespace fl
{


template<typename Activation, typename GateActivation, bool ReturnSequences>
LSTM<Activation, GateActivation, ReturnSequences>::LSTM(
        Eigen::Index input_len, Eigen::Index output_len,
        const Initializer<2> &initializer)
        : input_len(input_len),
          output_len(output_len)
{
    this->name = "lstm";
    this->w.resize(input_len + output_len, output_len * 4);

    // set W weights (top portion)
    this->w.slice(Dims<2>(0, 0), Dims<2>(input_len, output_len * 4))
            .device(this->device) = initializer(Dims<2>(input_len, output_len * 4),
                                                static_cast<int>(input_len),
                                                static_cast<int>(output_len));

    // set U weights (bottom portion)
    this->w.slice(Dims<2>(input_len, 0), Dims<2>(output_len, output_len * 4))
            .device(this->device) = initializer(Dims<2>(output_len, output_len * 4),
                                                static_cast<int>(output_len),
                                                static_cast<int>(output_len));

    this->dL_dw.resize(this->w.dimensions());
    this->dL_dw.setZero();
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
void LSTM<Activation, GateActivation, ReturnSequences>::Forward(
        const Tensor<3> &inputs)
{
    fl_assert(inputs.dimension(1) > 0,
              this->name << " expected an input with time dimension > 0");
    fl_assert(inputs.dimension(2) == this->input_len,
              this->name << " expected an input of "
                         << Dims<3>(-1, -1, this->input_len) << " got "
                         << inputs.dimensions() << " instead");

    this->h.resize(inputs.dimension(0), inputs.dimension(1), this->output_len);
    this->cs.resize(this->h.dimensions());

    if (this->lstm_cells.size() < inputs.dimension(1)) {
        int num_new_cells = inputs.dimension(1) - this->lstm_cells.size();

        for (int i = 0; i < num_new_cells; i++) {
            this->lstm_cells.push_back(
                    LSTMCell<Activation, GateActivation>(
                            this->lstm_cells.size(), this->input_len,
                            this->output_len));
        }
    }

    for (auto &cell: this->lstm_cells) {
        cell.Forward(inputs, this->w, this->h, this->cs, this->device);
    }

    if constexpr (!ReturnSequences) {
        this->h_no_seq.resize(inputs.dimension(0), this->output_len);
        this->h_no_seq.device(device) = this->h.chip(inputs.dimension(1) - 1, 1);
    }
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
void LSTM<Activation, GateActivation, ReturnSequences>::Forward(const Layer &prev)
{
    this->Forward(prev.GetOutput3D());
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
void LSTM<Activation, GateActivation, ReturnSequences>::Backward(
        const Tensor<ReturnSequences ? 3 : 2> &gradients)
{
    if constexpr(ReturnSequences) {
        this->dL_dw.setZero();
        Eigen::Index time_steps = this->h.dimension(1);

        this->lstm_cells.back().Backward(
                gradients.chip(time_steps - 1, 1),
                w, dL_dw,
                cs, fl::Tensor<2>(gradients.dimension(0), output_len).constant(0),
                device
        );

        for (auto i = time_steps - 2; i >= 0; i--) {
            lstm_cells[i].Backward(
                    gradients.chip(i, 1) +
                    this->lstm_cells[i + 1].GetInputGradientsHprev(),
                    w, dL_dw,
                    cs, lstm_cells[i + 1].GetInputGradientsCprev(),
                    device
            );
        }
    }
    else {
        Eigen::Index time_steps = this->h.dimension(1);

        this->dL_dw.setZero();
        this->lstm_cells.back().Backward(
                gradients, w, dL_dw,
                cs, fl::Tensor<2>(gradients.dimension(0), output_len).constant(0),
                device
        );

        for (auto i = time_steps - 2; i >= 0; i--) {
            lstm_cells[i].Backward(
                    lstm_cells[i + 1].GetInputGradientsHprev(), w, dL_dw,
                    cs, lstm_cells[i + 1].GetInputGradientsCprev(),
                    device
            );
        }
    }
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
void
LSTM<Activation, GateActivation, ReturnSequences>::Backward(Layer &next)
{
    if constexpr(ReturnSequences) {
        this->Backward(next.GetInputGradients3D());
    }
    else {
        this->Backward(next.GetInputGradients2D());
    }
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
void LSTM<Activation, GateActivation, ReturnSequences>::Update(Optimizer &optimizer)
{
    optimizer.Minimize(this->w, this->dL_dw);
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
const Tensor<2> &
LSTM<Activation, GateActivation, ReturnSequences>::GetOutput2D() const
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
LSTM<Activation, GateActivation, ReturnSequences>::GetOutput3D() const
{
    if constexpr(!ReturnSequences) {
        throw std::logic_error(
                this->name +
                " was initialized with ReturnSequences=false, use GetOutput2D instead");
    }

    return this->h;
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
std::vector<fl::Tensor<2>>
LSTM<Activation, GateActivation, ReturnSequences>::GetWeights2D() const
{
    return {this->w};
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
const Tensor<3> &
LSTM<Activation, GateActivation, ReturnSequences>::GetInputGradients3D()
{
    this->dL_dx.resize(this->h.dimension(0), this->h.dimension(1), this->input_len);
    int time_steps = static_cast<int>(this->h.dimension(1));

    for (int i = time_steps - 1; i >= 0; --i) {
        this->lstm_cells[i].CalcInputGradients(
                this->dL_dx, this->w, this->device);
    }

    return this->dL_dx;
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
std::vector<Tensor<2>>
LSTM<Activation, GateActivation, ReturnSequences>::GetWeightGradients2D() const
{
    return {this->dL_dw};
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
void LSTM<Activation, GateActivation, ReturnSequences>::SetWeights(
        const std::vector<fl::Tensor<2>> &weights)
{
    if (weights.size() != 2) {
        throw std::invalid_argument(
                this->name + " SetWeights() expects 2 tensors: weights and bias");
    }

    if (weights.front().size() != 0 &&
        weights.front().dimensions() !=
        this->w.dimensions()) {
        std::ostringstream error_msg;
        error_msg << this->name << " SetWeights() expects weight dimensions "
                  << this->w.dimensions() << ", got " << weights.front().dimensions()
                  << " instead";
        throw std::invalid_argument(error_msg.str());
    }
    else {
        this->w = weights.front();
    }
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
int LSTM<Activation, GateActivation, ReturnSequences>::GetInputRank() const
{
    return 3;
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
int LSTM<Activation, GateActivation, ReturnSequences>::GetOutputRank() const
{
    if constexpr(ReturnSequences) {
        return 3;
    }
    else {
        return 2;
    }
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
void LSTM<Activation, GateActivation, ReturnSequences>::Save(const std::string &path)
{
    std::ofstream output_file(path);
    output_file.precision(15);

    if (!output_file.is_open()) {
        throw std::invalid_argument(
                this->name + "::Save INVALID FILE PATH: " + path);
    }

    // flatten the weights and write it to the file with a white space delimiter
    Tensor<1> flatten = this->w.reshape(Dims<1>(this->w.size()));

    std::vector<Scalar> as_vector(flatten.data(), flatten.data() + flatten.size());
    std::copy(as_vector.begin(), as_vector.end(),
              std::ostream_iterator<Scalar>(output_file, " "));
    output_file.close();
}


template<typename Activation, typename GateActivation, bool ReturnSequences>
void LSTM<Activation, GateActivation, ReturnSequences>::Load(
        const std::string &path)
{
    std::ifstream read_weights(path);

    if (!read_weights.is_open()) {
        throw std::invalid_argument(this->name + "::Load " + path + " NOT FOUND");
    }

    std::vector<Scalar> as_vector;
    std::copy(std::istream_iterator<Scalar>(read_weights),
              std::istream_iterator<Scalar>(), std::back_inserter(as_vector));
    read_weights.close();

    if (as_vector.size() != this->w.size()) {
        std::ostringstream error_msg;
        error_msg << this->name << "::Load " << path << " EXPECTED "
                  << this->w.dimensions() << "=" << this->w.size() << " VALUES, GOT "
                  << as_vector.size() << " INSTEAD";
        throw std::invalid_argument(error_msg.str());
    }

    // reshape the flattened tensor back to expected weights dimensions
    this->w = TensorMap<2>(as_vector.data(), this->w.dimensions());
}


} // namespace fl
