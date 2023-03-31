//
// Created by RKuang on 8/29/2022.
//

#ifndef FLARE_SEQUENTIAL_HPP
#define FLARE_SEQUENTIAL_HPP

#include "flare/fl_types.hpp"
#include "flare/layers/layer.hpp"
#include "flare/loss/include_loss.hpp"
#include "flare/optimizers/include_optimizers.hpp"
#include <iomanip>

namespace fl
{

class Sequential
{
public:
    Sequential(std::initializer_list<Layer *> layers);


    Sequential() = default;


    void Add(Layer *layer);


    void ValidateLayers();


    template<int TensorSampleRank, int TensorLabelRank>
    void Fit(const std::vector<Tensor<TensorSampleRank>> &inputs,
             const std::vector<Tensor<TensorLabelRank>> &labels, int epochs,
             LossFunction<TensorLabelRank> &loss_function, Optimizer &opt);


    Layer &operator[](int layer_index);


    template<int OutputRank, int TensorSampleRank>
    const Tensor<OutputRank> &Predict(const Tensor<TensorSampleRank> &input,
                                      bool training_mode = false);

    template<int TensorSampleRank>
    void Forward(const Tensor<TensorSampleRank> &training_sample);

    template<int TensorLabelRank>
    void Backward(const Tensor<TensorLabelRank> &training_label,
                  LossFunction<TensorLabelRank> &loss_function);

    template<int TensorLabelRank>
    void Backward(const Tensor<TensorLabelRank> &gradients);

    void Update(Optimizer &optimizer);

    void Training(bool training);


    /**
     * Check that the layer weight gradients agree with the limit approximation to
     * epsilon'th degree of accuracy
     *
     * Training should stop after gradient checking since the loss history is
     * polluted with test values
     *
     * Gradient check can fail if vanishing or exploding gradients occur
     *
     * Currently works for Dense layers only
     * @param input   layer input tensor
     * @param label   expected output tensor
     * @param loss_function type of loss to use
     * @param epsilon accuracy to which gradients and limit approximation agree to
     * @return
     */
    Scalar GradientCheck(const Tensor<2> &input, const Tensor<2> &label,
                         LossFunction<2> &loss_function, Scalar epsilon = 1e-7);

    std::vector<Layer *> layers;

};


template<int OutputRank, int TensorSampleRank>
const Tensor<OutputRank> &Sequential::Predict(const Tensor<TensorSampleRank> &input,
                                              bool training_mode)
{
    this->Training(training_mode);
    this->Forward(input);

    if constexpr (OutputRank == 2) {
        return this->layers.back()->GetOutput2D();
    }
    else if constexpr (OutputRank == 3) {
        return this->layers.back()->GetOutput3D();
    }
    else {
        return this->layers.back()->GetOutput4D();
    }
}


template<int TensorSampleRank>
void Sequential::Forward(const Tensor<TensorSampleRank> &training_sample)
{
    this->layers.front()->Forward(training_sample);

    for (size_t i = 1; i < this->layers.size(); i++) {
        this->layers[i]->Forward(*this->layers[i - 1]);
    }
}


template<int TensorLabelRank>
void Sequential::Backward(const Tensor<TensorLabelRank> &training_label,
                          LossFunction<TensorLabelRank> &loss_function)
{
    if constexpr (TensorLabelRank == 2) {
        loss_function(this->layers.back()->GetOutput2D(), training_label);
    }
    else if constexpr (TensorLabelRank == 3) {
        loss_function(this->layers.back()->GetOutput3D(), training_label);
    }
    else if constexpr (TensorLabelRank == 4) {
        loss_function(this->layers.back()->GetOutput4D(), training_label);
    }
    else {
        throw std::logic_error("Sequential::Backward UNSUPPORTED TENSOR RANK " +
                               std::to_string(TensorLabelRank));
    }

    this->layers.back()->Backward(loss_function.GetGradients());

    for (int i = this->layers.size() - 2; i >= 0; --i) {
        this->layers[i]->Backward(*this->layers[i + 1]);
    }
}


template<int TensorSampleRank, int TensorLabelRank>
void Sequential::Fit(const std::vector<Tensor<TensorSampleRank>> &inputs,
                     const std::vector<Tensor<TensorLabelRank>> &labels,
                     int epochs, LossFunction<TensorLabelRank> &loss_function,
                     Optimizer &opt)
{
    if (inputs.size() != labels.size() || inputs.empty() || labels.empty()) {
        throw std::invalid_argument("inputs should match labels 1:1");
    }

    this->ValidateLayers();

    // to display progress bar
    const int num_batches = inputs.size();
    const int num_bars = 25;
    int batch_per_bar = num_batches / num_bars;
    int progress = 0;
    const int num_inputs = inputs.size();
    const int epoch_count_length = std::to_string(epochs).length();

    for (auto layer: this->layers) {
        layer->Training(true);
    }

    for (int e = 0; e < epochs; e++) {
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int m = 0; m < num_inputs; m++) {
            this->Forward(inputs[m]);
            this->Backward(labels[m], loss_function);
            this->Update(opt);

            // progress bar
            if (m % batch_per_bar == 0 && m != 0) {
                auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::high_resolution_clock::now() - start_time);

                // print behavior depends on terminal
                std::cout << "Epoch "
                          << std::setw(epoch_count_length) << std::setfill(' ')
                          << e + 1 << " ["
                          << std::setw(progress) << std::setfill('=') << ""
                          << std::setw(num_bars - progress)
                          << std::setfill('.') << "" << "] "
                          << elapsed_time.count() << "s loss: "
                          << loss_function.GetLoss() << "\r";
                std::cout.flush();
                progress++;
            }
        }

        progress = 0; // progress bar reset for next epoch
        std::cout << '\n';
    }

    for (auto layer: this->layers) {
        layer->Training(false);
    }
}


template<int TensorLabelRank>
void Sequential::Backward(const Tensor<TensorLabelRank> &gradients)
{
    this->layers.back()->Backward(gradients);

    for (int i = this->layers.size() - 2; i >= 0; --i) {
        this->layers[i]->Backward(*this->layers[i + 1]);
    }
}

}

#endif //FLARE_SEQUENTIAL_HPP
