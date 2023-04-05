//
// Created by R on 3/31/23.
//

#ifndef FLARE_BIDIRECTIONAL_HPP
#define FLARE_BIDIRECTIONAL_HPP

#include "lstm.hpp"

namespace fl
{

enum
{
    SUM, MEAN, HADAMARD, CONCAT,
    ADD = SUM, AVE = MEAN, MUL = HADAMARD
};

template<int Merge, typename Activation = TanH,
        typename GateActivation = Sigmoid,
        bool ReturnSequences = false>
class Bidirectional : public Layer
{
public:
    explicit Bidirectional(LSTM<Activation, GateActivation, ReturnSequences> *lstm);

    explicit Bidirectional(GRU<Activation, GateActivation, ReturnSequences> *gru);

    void Forward(const Tensor<3> &inputs) override;

    void Forward(const Layer &prev) override;

    void Backward(const Tensor<ReturnSequences ? 3 : 2> &gradients) override;

    void Backward(Layer &next) override;

    void Update(Optimizer &optimizer) override;

    const Tensor<2> &GetOutput2D() const override;

    const Tensor<3> &GetOutput3D() const override;

    const Tensor<3> &GetInputGradients3D() override;

    std::vector<Tensor<2>> GetWeights2D() const override;

    std::vector<Tensor<2>> GetWeightGradients2D() const override;

    void SetWeights(const std::vector<Tensor<2>> &weights) override;

    int GetInputRank() const override;

    int GetOutputRank() const override;


private:
    void SetSubLayerNames();

    inline void ForwardSum();

    inline void ForwardMean();

    inline void ForwardHadamard();

    inline void ForwardConcat();

    inline void BackwardSum(const Tensor<ReturnSequences ? 3 : 2> &gradients);

    inline void BackwardMean(const Tensor<ReturnSequences ? 3 : 2> &gradients);

    inline void BackwardHadamard(const Tensor<ReturnSequences ? 3 : 2> &gradients);

    inline void BackwardConcat(const Tensor<ReturnSequences ? 3 : 2> &gradients);

    Layer *forward_rnn;
    Layer *reverse_rnn;

    Dims<3, bool> rev_time_dim {false, true, false};
    Dims<3> input_dims;

    Tensor<ReturnSequences ? 3 : 2> h;
    Tensor<3> dL_dx;

    // multithreading
    Eigen::ThreadPool pool = Eigen::ThreadPool(
            (int) std::thread::hardware_concurrency());
    Eigen::ThreadPoolDevice device = Eigen::ThreadPoolDevice(&pool, 2);
};

} // namespace fl

#include "bidirectional.ipp"

#endif //FLARE_BIDIRECTIONAL_HPP
