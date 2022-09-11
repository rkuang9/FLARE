//
// Created by RKuang on 8/29/2022.
//

#include "sequential.h"

namespace orion {

    Sequential::Sequential(std::initializer_list<Layer *> layers) : layers(layers) {
        for (int i = 0; i < this->layers.size(); i++) {
            this->layers[i]->name += "_" + std::to_string(i + 1);
        }
    }


    void Sequential::Add(Layer *layer) {
        this->layers.push_back(layer);
    }


    void Sequential::Compile(Loss &loss_function, Optimizer &optimizer) {
        this->loss = &loss_function;
        this->opt = &optimizer;
    }


// currently accepts Tensor2D inputs (column vecors stacked side by side into a matrix)
    void Sequential::Fit(std::vector<std::vector<Scalar>> &inputs,
                         std::vector<std::vector<Scalar>> &labels,
                         int epochs, int batch_size) {
        if (inputs.size() != labels.size()) {
            throw std::invalid_argument("inputs should batch labels 1:1");
        }

        if (!this->loss) {
            throw std::logic_error("missing loss function");
        }

        if (!this->opt) {
            throw std::logic_error("missing optimizer");
        }

        this->training_data2D = orion::internal::VectorToBatch(inputs, batch_size);
        this->training_labels2D = orion::internal::VectorToBatch(labels,
                                                                 batch_size);

        for (int e = 0; e < epochs; e++) {
            for (int m = 0; m < this->training_data2D.size(); m++) {
                this->Forward(this->training_data2D[m]);
                this->Backward(this->training_labels2D[m], *this->loss);
                this->Update(*this->opt);
            }
        }
    }


    void Sequential::Forward(const Tensor2D &training_sample) {
        this->layers.front()->Forward(training_sample);

        for (size_t i = 1; i < this->layers.size(); i++) {
            this->layers[i]->Forward(*this->layers[i - 1]);
        }
    }


    void Sequential::Backward(const Tensor2D &training_label, Loss &loss_function) {
        loss_function.CalculateLoss(this->layers.back()->GetOutput2D(),
                                    training_label);

        this->layers.back()->Backward(loss_function);

        for (int i = this->layers.size() - 2; i >= 0; --i) {
            this->layers[i]->Backward(*this->layers[i + 1]);
        }
    }


    void Sequential::Update(Optimizer &optimizer) {
        for (Layer *layer: this->layers) {
            layer->Update(*this->opt);
        }
    }


    Layer &Sequential::operator[](int layer_index) {
        return *this->layers[layer_index];
    }


    Tensor<2> Sequential::Predict(const Tensor<2> &example) {
        this->Forward(example);
        return this->layers.back()->GetOutput2D();
    }


    Scalar Sequential::GradientCheck(const Tensor<2> &input, const Tensor<2> &label,
                                     Scalar epsilon) {
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
                    Scalar J_plus = (*this->loss)(
                            this->layers.back()->GetOutput2D(), label);

                    // J(0...0-E...0) term, 2* to undo J_plus too
                    theta(row, col) -= 2 * epsilon;
                    this->layers[i]->SetWeights(theta);
                    this->Forward(input);
                    Scalar J_minus = (*this->loss)(
                            this->layers.back()->GetOutput2D(), label);

                    dtheta_approx(row, col) = (J_plus - J_minus) / (2 * epsilon);

                    // restore original layer weights
                    this->layers[i]->SetWeights(original_weights);
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

        Tensor<0> result = nominator.sqrt() / (denominator_left.sqrt() + denominator_right.sqrt());
        return result(0);
    }

} // namespace orion