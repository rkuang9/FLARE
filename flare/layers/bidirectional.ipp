//
// Created by R on 3/31/23.
//

#include "bidirectional.hpp"

namespace fl
{


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::Bidirectional(
        LSTM <Activation, GateActivation, ReturnSequences> *lstm)
        : forward_rnn(lstm),
          reverse_rnn(new LSTM<Activation, GateActivation, ReturnSequences>(*lstm))
{
    this->SetSubLayerNames();
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::Bidirectional(
        GRU <Activation, GateActivation, ReturnSequences> *gru)
        : forward_rnn(gru),
          reverse_rnn(new GRU<Activation, GateActivation, ReturnSequences>(*gru))
{
    this->SetSubLayerNames();
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
void Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::Forward(
        const Tensor<3> &inputs)
{
    this->input_dims = inputs.dimensions();
    this->forward_rnn->Forward(inputs);
    this->reverse_rnn->Forward(Tensor<3>(inputs.reverse(this->rev_time_dim)));

    // resize for multi-threading
    if constexpr(ReturnSequences) {
        if constexpr(Merge == CONCAT) {
            this->h.resize(this->forward_rnn->GetOutput3D().dimension(0),
                           this->forward_rnn->GetOutput3D().dimension(1),
                           this->forward_rnn->GetOutput3D().dimension(2) * 2);
        }
        else {
            this->h.resize(this->forward_rnn->GetOutput3D().dimensions());
        }
    }
    else {
        if constexpr(Merge == CONCAT) {
            this->h.resize(this->forward_rnn->GetOutput2D().dimension(0),
                           this->forward_rnn->GetOutput2D().dimension(1) * 2);
        }
        else {
            this->h.resize(this->forward_rnn->GetOutput2D().dimensions());
        }
    }

    if constexpr(Merge == SUM) {
        this->ForwardSum();
    }
    else if constexpr(Merge == HADAMARD) {
        this->ForwardHadamard();
    }
    else if constexpr(Merge == MEAN) {
        this->ForwardMean();
    }
    else if constexpr(Merge == CONCAT) {
        this->ForwardConcat();
    }
    else {
        throw std::invalid_argument(
                this->name + " Forward() invalid merge template parameter");
    }
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
void Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::Forward(
        const Layer &prev)
{
    if constexpr(ReturnSequences) {
        this->Forward(prev.GetOutput3D());
    }
    else {
        this->Forward(prev.GetOutput2D());
    }
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
void Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::Backward(
        const Tensor<ReturnSequences ? 3 : 2> &gradients)
{
    if constexpr(Merge == SUM) {
        this->BackwardSum(gradients);
    }
    else if constexpr(Merge == HADAMARD) {
        this->BackwardHadamard(gradients);
    }
    else if constexpr(Merge == MEAN) {
        this->BackwardMean(gradients);
    }
    else if constexpr(Merge == CONCAT) {
        this->BackwardConcat(gradients);
    }
    else {
        throw std::invalid_argument(
                this->name + " Backward() invalid merge template parameter");
    }
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
void
Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::Backward(
        Layer &next)
{
    if constexpr(ReturnSequences) {
        this->Backward(next.GetInputGradients3D());
    }
    else {
        this->Backward(next.GetInputGradients2D());
    }
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
void Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::Update(
        Optimizer &optimizer)
{
    this->forward_rnn->Update(optimizer);
    this->reverse_rnn->Update(optimizer);
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
const Tensor<2> &
Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::GetOutput2D() const
{
    if constexpr(ReturnSequences) {
        throw std::logic_error(
                this->name +
                " was initialized with ReturnSequences=true, use GetOutput3D instead");
    }

    return this->h;
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
const Tensor<3> &
Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::GetOutput3D() const
{
    if constexpr(!ReturnSequences) {
        throw std::logic_error(
                this->name +
                " was initialized with ReturnSequences=false, use GetOutput2D instead");
    }

    return this->h;
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
const Tensor<3> &
Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::GetInputGradients3D()
{
    this->dL_dx.resize(this->input_dims);

    this->dL_dx.device(this->device) =
            this->forward_rnn->GetInputGradients3D() +
            this->reverse_rnn->GetInputGradients3D().reverse(this->rev_time_dim);

    return this->dL_dx;
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
std::vector<fl::Tensor<2>>
Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::GetWeights2D() const
{
    auto fwd_dw = this->forward_rnn->GetWeights2D();
    auto rev_dw = this->reverse_rnn->GetWeights2D();

    fwd_dw.insert(fwd_dw.end(), rev_dw.begin(), rev_dw.end());

    return fwd_dw;
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
void Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::SetWeights(
        const std::vector<fl::Tensor<2>> &weights)
{
    // {w_u_weights, bias, w_u_weights, bias}
    if (weights.size() != 4) {
        throw std::invalid_argument(
                this->name +
                " SetWeights() expects 4 values: 2 sets of weights & bias tensors");
    }

    this->forward_rnn->SetWeights(std::vector<Tensor<2 >> {weights[0], weights[1]});
    this->reverse_rnn->SetWeights(std::vector<Tensor<2>> {weights[2], weights[3]});
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
std::vector<fl::Tensor<2>>
Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::GetWeightGradients2D() const
{
    auto fwd_dw = this->forward_rnn->GetWeightGradients2D();
    auto rev_dw = this->reverse_rnn->GetWeightGradients2D();

    fwd_dw.insert(fwd_dw.end(), rev_dw.begin(), rev_dw.end());

    return fwd_dw;
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
int
Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::GetInputRank() const
{
    return 3;
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
int
Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::GetOutputRank() const
{
    if constexpr(ReturnSequences) {
        return 3;
    }
    else {
        return 2;
    }
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
void
Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::SetSubLayerNames()
{
    this->name = "bidirectional_" + this->forward_rnn->name;
    this->forward_rnn->name += "_left_right";
    this->reverse_rnn->name += "_right_left";
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
void Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::ForwardSum()
{
    if constexpr(ReturnSequences) {
        this->h.device(this->device) = this->forward_rnn->GetOutput3D() +
                                       this->reverse_rnn->GetOutput3D()
                                               .reverse(this->rev_time_dim);
    }
    else {
        this->h.device(this->device) = this->forward_rnn->GetOutput2D() +
                                       this->reverse_rnn->GetOutput2D();
    }
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
void
Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::ForwardMean()
{
    if constexpr(ReturnSequences) {
        this->h.device(this->device) = (this->forward_rnn->GetOutput3D() +
                                        this->reverse_rnn->GetOutput3D()
                                                .reverse(this->rev_time_dim)) * 0.5;
    }
    else {
        this->h.device(this->device) = (this->forward_rnn->GetOutput2D() +
                                        this->reverse_rnn->GetOutput2D()) * 0.5;
    }
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
void
Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::ForwardHadamard()
{
    if constexpr(ReturnSequences) {
        this->h.device(this->device) = this->forward_rnn->GetOutput3D() *
                                       this->reverse_rnn->GetOutput3D()
                                               .reverse(this->rev_time_dim);
    }
    else {
        this->h.device(this->device) = this->forward_rnn->GetOutput2D() *
                                       this->reverse_rnn->GetOutput2D();
    }
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
void
Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::ForwardConcat()
{
    if constexpr(ReturnSequences) {
        this->h.device(this->device) = this->forward_rnn->GetOutput3D()
                .concatenate(this->reverse_rnn->GetOutput3D()
                                     .reverse(this->rev_time_dim), 2);
    }
    else {
        this->h.device(this->device) = this->forward_rnn->GetOutput2D()
                .concatenate(this->reverse_rnn->GetOutput2D(), 1);
    }
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
void Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::BackwardSum(
        const Tensor<ReturnSequences ? 3 : 2> &gradients)
{
    if constexpr(ReturnSequences) {
        this->forward_rnn->Backward(gradients);
        this->reverse_rnn->Backward(
                Tensor<3>(gradients.reverse(this->rev_time_dim)));
    }
    else {
        this->forward_rnn->Backward(gradients);
        this->reverse_rnn->Backward(gradients);
    }
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
void
Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::BackwardMean(
        const Tensor<ReturnSequences ? 3 : 2> &gradients)
{
    if constexpr(ReturnSequences) {
        this->forward_rnn->Backward(Tensor<3>(0.5 * gradients));
        this->reverse_rnn->Backward(
                Tensor<3>(0.5 * gradients.reverse(this->rev_time_dim)));
    }
    else {
        this->forward_rnn->Backward(Tensor<2>(0.5 * gradients));
        this->reverse_rnn->Backward(Tensor<2>(0.5 * gradients));
    }
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
void
Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::BackwardHadamard(
        const Tensor<ReturnSequences ? 3 : 2> &gradients)
{
    if constexpr(ReturnSequences) {
        this->forward_rnn->Backward(Tensor<3>(
                gradients *
                this->reverse_rnn->GetOutput3D().reverse(this->rev_time_dim)));

        // TODO: TensorFlow has a different output for reverse_rnn weight grads, not sure which is wrong
        this->reverse_rnn->Backward(
                Tensor<3>(gradients * this->forward_rnn->GetOutput3D()));
    }
    else {
        this->forward_rnn->Backward(
                Tensor<2>(gradients * this->reverse_rnn->GetOutput2D()));
        this->reverse_rnn->Backward(
                Tensor<2>(gradients * this->forward_rnn->GetOutput2D()));
    }
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
void
Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::BackwardConcat(
        const Tensor<ReturnSequences ? 3 : 2> &gradients)
{
    if constexpr(ReturnSequences) {
        Dims<3> fwd_offset(0, 0, 0);
        Dims<3> rev_offset(0, 0, this->h.dimension(2) / 2);
        Dims<3> extent(gradients.dimension(0), gradients.dimension(1),
                       this->h.dimension(2) / 2);

        this->forward_rnn->Backward(Tensor<3>(gradients.slice(fwd_offset, extent)));
        this->reverse_rnn->Backward(Tensor<3>(
                gradients.slice(rev_offset, extent).reverse(this->rev_time_dim)));
    }
    else {
        Dims<2> fwd_offset(0, 0);
        Dims<2> rev_offset(0, this->h.dimension(1) / 2);
        Dims<2> extent(gradients.dimension(0), this->h.dimension(1) / 2);

        this->forward_rnn->Backward(Tensor<2>(gradients.slice(fwd_offset, extent)));
        this->reverse_rnn->Backward(Tensor<2>(gradients.slice(rev_offset, extent)));
    }
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
void Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::Save(
        const std::string &path)
{
    this->forward_rnn->Save(path + ".bi_rnn_fwd");
    this->reverse_rnn->Save(path + ".bi_rnn_rev");
}


template<int Merge, typename Activation, typename GateActivation, bool ReturnSequences>
void Bidirectional<Merge, Activation, GateActivation, ReturnSequences>::Load(
        const std::string &path)
{
    this->forward_rnn->Load(path + ".bi_rnn_fwd");
    this->reverse_rnn->Load(path + ".bi_rnn_rev");
}


} // namespace fl