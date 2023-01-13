//
// Created by RKuang on 8/29/2022.
//

#include "sequential.hpp"
#include <iomanip>
#include <iostream>

namespace orion
{

Sequential::Sequential(std::initializer_list<Layer *> layers) : layers(layers)
{
    for (int i = 0; i < this->layers.size(); i++) {
        this->layers[i]->name += "_" + std::to_string(i);
    }
}


void Sequential::Add(Layer *layer)
{
    layer->name += "_" + std::to_string(this->layers.size());
    this->layers.push_back(layer);
}


void Sequential::Compile(LossFunction &loss_function, Optimizer &optimizer)
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
        throw std::logic_error("Sequential::Compile FOUND INCOMPATIBLE LAYERS\n" +
                               error_msg.str());
    }

    this->loss = &loss_function;
    this->opt = &optimizer;
}


void Sequential::Update(Optimizer &optimizer)
{
    for (Layer *layer: this->layers) {
        // update each layer's learnable parameters
        layer->Update(*this->opt);
    }

    // let the optimizer know to move to the next iteration t
    // since some optimizers need to update their internal parameters
    optimizer.Step();
}


Layer &Sequential::operator[](int layer_index)
{
    return *this->layers[layer_index];
}


Scalar Sequential::GradientCheck(const Tensor<2> &input, const Tensor<2> &label,
                                 Scalar epsilon)
{
    // perform the first training pass to compute dL/dw (without updating)
    this->Forward(input);
    this->Backward(label, *this->loss);

    std::vector<Tensor<2>> approx_gradients;
    std::vector<Tensor<2>> actual_gradients;

    // perform gradient check one layer at a time
    for (int i = 0; i < this->layers.size(); i++) {
        if (this->layers[i]->GetWeights().size() == 0) {
            continue; // skip layers that don't have weights
        }

        actual_gradients.push_back(this->layers[i]->GetWeightGradients());

        // to restore layer weights
        Tensor<2> original_weights = this->layers[i]->GetWeights();

        // temporary weights modified with epsilon
        Tensor<2> theta = this->layers[i]->GetWeights();

        // built using limit definition double-sided difference of derivatives
        Tensor<2> dtheta_approx(original_weights.dimensions());

        // for each element do a forward pass with the target weight element
        // modified with a tiny epsilon value, calculate the formula
        // approx dL/dw = (J(0...0+E...0) - J(0...0-E...0)) / 2E
        for (Eigen::Index col = 0; col < theta.dimension(1); col++) {
            for (Eigen::Index row = 0; row < theta.dimension(0); row++) {
                // J(0...0+E...0) term
                theta(row, col) += epsilon;
                this->layers[i]->SetWeights(theta);
                this->Forward(input);
                this->loss->CalculateLoss(this->layers.back()->GetOutput2D(), label);
                Scalar J_plus = this->loss->GetLoss();

                // J(0...0-E...0) term, 2* to undo J_plus too
                theta(row, col) -= 2 * epsilon;
                this->layers[i]->SetWeights(theta);
                this->Forward(input);
                this->loss->CalculateLoss(this->layers.back()->GetOutput2D(), label);
                Scalar J_minus = this->loss->GetLoss();

                dtheta_approx(row, col) = (J_plus - J_minus) / (2 * epsilon);

                // restore original layer weights and theta
                this->layers[i]->SetWeights(original_weights);
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
}

} // namespace orion