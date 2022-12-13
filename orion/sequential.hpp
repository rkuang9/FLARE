//
// Created by RKuang on 8/29/2022.
//

#ifndef ORION_SEQUENTIAL_HPP
#define ORION_SEQUENTIAL_HPP

#include "orion/orion_types.hpp"
//#include "orion/layers/include_layers.hpp"
#include "orion/layers/layer.hpp"
#include "orion/loss/include_loss.hpp"
#include "orion/optimizers/include_optimizers.hpp"
#include <iomanip>

namespace orion
{

class Sequential
{
public:
    Sequential(std::initializer_list<Layer *> layers);

    void Add(Layer *layer);

    void Compile(LossFunction &loss_function, Optimizer &optimizer);

    template<int TensorSampleRank, int TensorLabelRank>
    void Fit(const std::vector<Tensor<TensorSampleRank>> &inputs,
             const std::vector<Tensor<TensorLabelRank>> &labels,
             int epochs)
    {
        if (inputs.size() != labels.size() || inputs.empty() || labels.empty()) {
            throw std::invalid_argument("inputs should match labels 1:1");
        }

        if (!this->loss) {
            throw std::logic_error("missing loss function");
        }

        if (!this->opt) {
            throw std::logic_error("missing optimizer");
        }

        this->total_samples = inputs.size();

        // to display progress bar
        int num_batches = inputs.size();
        int num_bars = 25;
        int batch_per_bar = num_batches / num_bars;
        int progress = 0;
        int num_inputs = inputs.size();
        int epoch_count_length = std::to_string(epochs).length();

        for (int e = 0; e < epochs; e++) {
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int m = 0; m < num_inputs; m++) {
                this->Forward(inputs[m]);
                this->Backward(labels[m], *this->loss);
                this->Update(*this->opt);

                // progress bar
                if (m % batch_per_bar == 0 && m != 0) {

                    auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::high_resolution_clock::now() - start_time);

                    // progress bar, if ran in CLion, it will print
                    // new lines per iteration due to the PuTTY terminal
                    std::cout << "\rEpoch "
                              << std::setw(epoch_count_length) << std::setfill(' ')
                              << e + 1 << " ["
                              << std::setw(progress + 1) << std::setfill('=') << '>'
                              << std::setw(num_bars - progress)
                              << std::setfill('.') << "] "
                              << elapsed_time.count() << "s loss: "
                              << std::setprecision(5) << this->loss->GetLoss();
                    progress++;
                }
            }

            progress = 0; // progress bar reset for next epoch
            std::cout << '\n';

        }
    }


    Layer &operator[](int layer_index);

    Tensor<2> Predict(const Tensor<2> &example);

    template<int OutputRank, int TensorSampleRank>
    Tensor<OutputRank> Predict(const Tensor<TensorSampleRank> &example)
    {
        this->Forward(example);

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
     * @param epsilon accuracy to which gradients and limit approximation agree to
     * @return
     */
    Scalar GradientCheck(const Tensor<2> &input, const Tensor<2> &label,
                         Scalar epsilon = 1e-7);

    std::vector<Layer *> layers;

    const LossFunction *GetLossFunction() const;

    int GetTotalSamples() const;

protected:
    template<int TensorSampleRank>
    void Forward(const Tensor<TensorSampleRank> &training_sample)
    {
        this->layers.front()->Forward(training_sample);

        for (size_t i = 1; i < this->layers.size(); i++) {
            this->layers[i]->Forward(*this->layers[i - 1]);
        }
    }

    template<int TensorLabelRank>
    void Backward(const Tensor<TensorLabelRank> &training_label,
                  LossFunction &loss_function)
    {

        if constexpr (TensorLabelRank == 2) {
            loss_function.CalculateLoss(this->layers.back()->GetOutput2D(),
                                        training_label);
        }
        else if constexpr (TensorLabelRank == 3) {
            loss_function.CalculateLoss(this->layers.back()->GetOutput3D(),
                                        training_label);
        }
        else if constexpr (TensorLabelRank == 4) {
            loss_function.CalculateLoss(this->layers.back()->GetOutput4D(),
                                        training_label);
        }

        this->layers.back()->Backward(loss_function);

        for (int i = this->layers.size() - 2; i >= 0; --i) {
            this->layers[i]->Backward(*this->layers[i + 1]);
        }
    }

    void Update(Optimizer &optimizer);


    std::vector<Tensor<2>> training_data2D;
    std::vector<Tensor<2>> training_labels2D;

    LossFunction *loss = nullptr;
    Optimizer *opt = nullptr;
    int total_samples = 0;

};

}

#endif //ORION_SEQUENTIAL_HPP
