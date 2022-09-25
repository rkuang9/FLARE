//
// Created by RKuang on 8/29/2022.
//

#ifndef ORION_SEQUENTIAL_HPP
#define ORION_SEQUENTIAL_HPP

#include "orion/orion_types.hpp"
#include "orion/layers/include_layers.hpp"
#include "orion/loss/include_loss.hpp"
#include "orion/optimizers/include_optimizers.hpp"
#include "orion/internal/batcher.hpp"

namespace orion
{

class Sequential
{
public:
    Sequential(std::initializer_list<Layer *> layers);

    void Add(Layer *layer);

    void Compile(Loss &loss_function, Optimizer &optimizer);

    void Fit(std::vector<std::vector<Scalar>> &inputs,
             std::vector<std::vector<Scalar>> &labels,
             int epochs, int batch_size = 1);

    void Fit(const std::vector<Tensor<2>> &inputs,
             const std::vector<Tensor<2>> &labels,
             int epochs, int batch_size = 1);

    Layer &operator[](int layer_index);

    Tensor<2> Predict(const Tensor<2> &example);

    Scalar GradientCheck(const Tensor<2> &input, const Tensor<2> &label,
                       Scalar epsilon = 1e-7);

    std::vector<Layer *> layers;
protected:

    void Forward(const Tensor2D &training_sample);

    void Backward(const Tensor2D &training_label, Loss &loss_function);

    void Update(Optimizer &optimizer);


    std::vector<Tensor2D> training_data2D;
    std::vector<Tensor2D> training_labels2D;

    Loss *loss = nullptr;
    Optimizer *opt = nullptr;
    int epochs = 0; // not in use
    int batch_size = 32; // not in use
};

}

#endif //ORION_SEQUENTIAL_HPP
