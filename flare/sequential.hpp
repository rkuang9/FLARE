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
    Sequential(std::initializer_list<Layer *> layers) : layers(layers)
    {
        for (int i = 0; i < this->layers.size(); i++) {
            this->layers[i]->name += "_" + std::to_string(i);
        }
    }


    Sequential() = default;


    void Add(Layer *layer)
    {
        layer->name += "_" + std::to_string(this->layers.size());
        this->layers.push_back(layer);
    }


    void ValidateLayers()
    {
        if (this->layers.empty()) {
            throw std::logic_error("Sequential::Compile NO LAYERS");
        }

        bool layers_compatible = true;
        std::vector<std::string> incompatible_layers;

        for (int i = 1; i < this->layers.size(); i++) {
            if (this->layers[i]->GetInputRank() !=
                this->layers[i - 1]->GetOutputRank()) {
                layers_compatible = false;
                incompatible_layers.push_back(this->layers[i - 1]->name +
                                              " -> " + this->layers[i]->name);
            }
        }

        if (!layers_compatible) {
            std::ostringstream error_msg;
            std::copy(incompatible_layers.begin(), incompatible_layers.end(),
                      std::ostream_iterator<std::string>(error_msg, "\n"));
            throw std::logic_error(
                    "Sequential::Compile FOUND INCOMPATIBLE LAYERS\n" +
                    error_msg.str());
        }
    }


    template<int TensorSampleRank, int TensorLabelRank>
    void Fit(const std::vector<fl::Tensor<TensorSampleRank>> &inputs,
             const std::vector<fl::Tensor<TensorLabelRank>> &labels, int epochs,
             LossFunction <TensorLabelRank> &loss_function, Optimizer &opt,
             const std::vector<fl::Metric<TensorLabelRank> *> &metrics = {})
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

                for (auto metric: metrics) {
                    if constexpr(TensorLabelRank == 2) {
                        (*metric)(labels[m], this->layers.back()->GetOutput2D());
                    }
                    else if constexpr(TensorLabelRank == 3) {
                        (*metric)(labels[m], this->layers.back()->GetOutput3D());
                    }
                    else {
                        (*metric)(labels[m], this->layers.back()->GetOutput4D());
                    }
                }

                // progress bar
                if (m % batch_per_bar == 0 && m != 0) {
                    auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::high_resolution_clock::now() - start_time);

                    // print behavior depends on terminal
                    std::cout << "\rEpoch "
                              << std::setw(epoch_count_length) << std::setfill(' ')
                              << e + 1 << " ["
                              << std::setw(progress) << std::setfill('=') << ""
                              << std::setw(num_bars - progress)
                              << std::setfill('.') << "" << "] "
                              << elapsed_time.count() << "s loss: "
                              << std::setprecision(4) << std::fixed
                              << loss_function.GetLoss();

                    for (auto metric: metrics) {
                        std::cout << ", " << *metric;
                    }

                    std::cout << std::flush;
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


    Layer &operator[](int layer_index)
    {
        return *this->layers[layer_index];
    }


    template<int OutputRank, int TensorSampleRank>
    const Tensor <OutputRank> &Predict(const Tensor <TensorSampleRank> &input,
                                       bool training_mode = false)
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
    void Forward(const Tensor <TensorSampleRank> &training_sample)
    {
        this->layers.front()->Forward(training_sample);

        for (size_t i = 1; i < this->layers.size(); i++) {
            this->layers[i]->Forward(*this->layers[i - 1]);
        }
    }


    template<int TensorLabelRank>
    void Backward(const Tensor <TensorLabelRank> &training_label,
                  LossFunction <TensorLabelRank> &loss_function)
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


    template<int TensorLabelRank>
    void Backward(const Tensor <TensorLabelRank> &gradients)
    {
        this->layers.back()->Backward(gradients);

        for (int i = this->layers.size() - 2; i >= 0; --i) {
            this->layers[i]->Backward(*this->layers[i + 1]);
        }
    }


    void Update(Optimizer &optimizer)
    {
        for (Layer *layer: this->layers) {
            // update each layer's learnable parameters
            layer->Update(optimizer);
        }

        // let the optimizer know to move to the next iteration t
        // since some optimizers need to update their internal parameters
        optimizer.Step();
    }


    void Training(bool training)
    {
        for (auto layer: this->layers) {
            layer->Training(training);
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
     * @param loss_function type of loss to use
     * @param epsilon accuracy to which gradients and limit approximation agree to
     * @return
     */
    /*Scalar GradientCheck(const Tensor<2> &input, const Tensor<2> &label,
                         LossFunction<2> &loss_function, Scalar epsilon = 1e-7)
    {
        // perform the first training pass to compute dL/dw (without updating)
        this->Forward(input);
        this->Backward(label, loss_function);

        std::vector<Tensor<2>> approx_gradients;
        std::vector<Tensor<2>> actual_gradients;

        // perform gradient check one layer at a time
        for (int i = 0; i < this->layers.size(); i++) {
            if (this->layers[i]->GetWeights2D().size() == 0) {
                continue; // skip layers that don't have weights
            }

            actual_gradients.push_back(
                    this->layers[i]->GetWeightGradients2D().front());

            // to restore layer weights
            Tensor<2> original_weights = this->layers[i]->GetWeights2D().front();

            // temporary weights modified with epsilon
            Tensor<2> theta = this->layers[i]->GetWeights2D().front();

            // built using limit definition double-sided difference of derivatives
            Tensor<2> dtheta_approx(original_weights.dimensions());

            // for each element do a forward pass with the target weight element
            // modified with a tiny epsilon value, calculate the formula
            // approx dL/dw = (J(0...0+E...0) - J(0...0-E...0)) / 2E
            for (Eigen::Index col = 0; col < theta.dimension(1); col++) {
                for (Eigen::Index row = 0; row < theta.dimension(0); row++) {
                    // J(0...0+E...0) term
                    theta(row, col) += epsilon;
                    this->layers[i]->SetWeights(std::vector<Tensor<2>> {theta});
                    this->Forward(input);
                    loss_function(this->layers.back()->GetOutput2D(), label);
                    Scalar J_plus = loss_function.GetLoss();

                    // J(0...0-E...0) term, 2* to undo J_plus too
                    theta(row, col) -= 2 * epsilon;
                    this->layers[i]->SetWeights(std::vector<Tensor<2>> {theta});
                    this->Forward(input);
                    loss_function(this->layers.back()->GetOutput2D(), label);
                    Scalar J_minus = loss_function.GetLoss();

                    dtheta_approx(row, col) = (J_plus - J_minus) / (2 * epsilon);

                    // restore original layer weights and theta
                    this->layers[i]->SetWeights(
                            std::vector<Tensor<2>> {original_weights});
                    theta(row, col) += epsilon; // undo epsilon used for J+ & J-
                }
            }

            approx_gradients.push_back(dtheta_approx);
        }

        // apply L2norm(d0_approx - d0) / L2norm(d0_approx) + L2norm(d0)
        Tensor<0> nominator;
        nominator.setValues(0);
        Tensor<0> denominator_left, denominator_right;
        denominator_left.setValues(0);
        denominator_right.setValues(0);

        for (int i = 0; i < actual_gradients.size(); i++) {
            nominator += (actual_gradients[i] - approx_gradients[i]).square().sum();
            denominator_left += approx_gradients[i].square().sum();
            denominator_right += actual_gradients[i].square().sum();
        }

        Tensor<0> result = nominator.sqrt() /
                           (denominator_left.sqrt() + denominator_right.sqrt());
        return result(0);
    }*/


    std::vector<Layer *> layers;

};

}

#endif //FLARE_SEQUENTIAL_HPP
