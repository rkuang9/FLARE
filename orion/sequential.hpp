//
// Created by RKuang on 8/29/2022.
//

#ifndef ORION_SEQUENTIAL_HPP
#define ORION_SEQUENTIAL_HPP

#include "orion/orion_types.hpp"
#include "orion/layers/include_layers.hpp"
#include "orion/loss/include_loss.hpp"
#include "orion/optimizers/include_optimizers.hpp"
#include "orion/internal/include_internal.hpp"
#include "orion/metrics/include_metrics.hpp"
#include <iomanip>

namespace orion
{

class Sequential
{
public:
    Sequential(std::initializer_list<Layer *> layers);

    void Add(Layer *layer);

    void Compile(LossFunction &loss_function, Optimizer &optimizer);


    /*void Fit(std::vector<std::vector<Scalar>> &inputs,
             std::vector<std::vector<Scalar>> &labels,
             int epochs, int batch_size = 1);

    void Fit(const std::vector<Tensor<2>> &inputs,
             const std::vector<Tensor<2>> &labels,
             int epochs, int batch_size = 1);*/

    template<int TensorSampleRank, int TensorLabelRank>
    void Fit(const std::vector<Tensor<TensorSampleRank>> &inputs,
             const std::vector<Tensor<TensorLabelRank>> &labels,
             int epochs, int batch_size = 1)
    {
        std::cout << "running fit\n";

        if (inputs.size() != labels.size()) {
            throw std::invalid_argument("inputs should match labels 1:1");
        }

        if (!this->loss) {
            throw std::logic_error("missing loss function");
        }

        if (!this->opt) {
            throw std::logic_error("missing optimizer");
        }

        this->epochs = epochs;
        this->batch_size = batch_size;
        this->total_samples = inputs.size();

        for (int e = 0; e < epochs; e++) {
            for (int m = 0; m < inputs.size(); m++) {
                this->Forward(inputs[m]);
                this->Backward(labels[m], *this->loss);
                this->Update(*this->opt);

                // print metrics on screen
                if (!this->metrics.empty()) {
                    std::cout << "Epoch " << std::right
                            << std::setw(std::to_string(epochs).length()) << e + 1;

                    for (auto *metric: this->metrics) {
                        std::cout << " - " << metric->name << ": "
                                << metric->Compute(*this)
                                << " ";
                    }

                    std::cout << "\n";
                }
            }
        }
    }


    Layer &operator[](int layer_index);

    Tensor<2> Predict(const Tensor<2> &example);

    /**
     * Check that the layer weight gradients agree with the limit approximation to
     * epsilon'th degree of accuracy
     *
     * Training should stop after gradient checking since the loss history is
     * polluted with test values
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

    int GetEpochs() const;

    int GetBatchSize() const;

    int GetTotalSamples() const;

    void Compile(LossFunction &loss_function, Optimizer &optimizer,
                 const std::vector<Metric *> &metrics);

protected:
    template <int TensorSampleRank>
    void Forward(const Tensor<TensorSampleRank> &training_sample)
    {
        this->layers.front()->Forward(training_sample);

        for (size_t i = 1; i < this->layers.size(); i++) {
            this->layers[i]->Forward(*this->layers[i - 1]);
        }
    }

    template <int TensorLabelRank>
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
    int epochs = 0; // not in use
    int batch_size = 32; // not in use
    int total_samples = 0;
    std::vector<Metric *> metrics;

};

}

#endif //ORION_SEQUENTIAL_HPP
