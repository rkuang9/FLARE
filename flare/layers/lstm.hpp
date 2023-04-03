//
// Created by R on 3/25/23.
//

#ifndef FLARE_LSTM_HPP
#define FLARE_LSTM_HPP

#include "layer.hpp"
#include "rnn/lstm_cell.hpp"

namespace fl
{

template<typename Activation = TanH,
        typename GateActivation = Sigmoid,
        bool ReturnSequences = false>
class LSTM : public Layer
{
public:
    LSTM(Eigen::Index input_len, Eigen::Index output_len,
         const Initializer<2> &initializer = GlorotUniform<2>());

    void Forward(const Tensor<3> &inputs) override;

    void Forward(const Layer &prev) override;

    //void Backward(const Tensor<2> &gradients) override;

    //void Backward(const Tensor<3> &gradients) override;

    void Backward(const Tensor<ReturnSequences ? 3 : 2> &gradients) override;

    void Backward(Layer &next) override;

    void Update(Optimizer &optimizer) override;

    const Tensor<2> &GetOutput2D() const override;

    const Tensor<3> &GetOutput3D() const override;

    const Tensor<3> &GetInputGradients3D() override;

    const Tensor<2> &GetWeightGradients() const override;

    std::vector<Tensor<2>> GetWeights2D() const override;

    void SetWeights(const std::vector<Tensor<2>> &weights) override;

    void Save(const std::string &path) override;

    void Load(const std::string &path) override;

    int GetInputRank() const override;

    int GetOutputRank() const override;

private:
    Eigen::Index input_len;
    Eigen::Index output_len;

    //Tensor<3> x; // held in LSTMCells instead
    Tensor<3> dL_dx;

    Tensor<3> h;
    Tensor<3> cs;

    Tensor<2> h_no_seq; // holds the last cell's output for ReturnSequences == true

    Tensor<2> w;
    Tensor<2> dL_dw;

    std::vector<LSTMCell<Activation, GateActivation>> lstm_cells;

    // multithreading
    Eigen::ThreadPoolDevice device = Eigen::ThreadPoolDevice(new Eigen::ThreadPool(
            (int) std::thread::hardware_concurrency()), 2);
};

} // namespace fl

#include "lstm.ipp"

#endif //FLARE_LSTM_HPP





