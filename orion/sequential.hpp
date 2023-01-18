//
// Created by RKuang on 8/29/2022.
//

#ifndef ORION_SEQUENTIAL_HPP
#define ORION_SEQUENTIAL_HPP

#include "orion/orion_types.hpp"
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


    Sequential() = default;


    void Add(Layer *layer);


    void Compile(LossFunction &loss_function, Optimizer &optimizer);


    template<int TensorSampleRank, int TensorLabelRank>
    void Fit(const std::vector<Tensor<TensorSampleRank>> &inputs,
             const std::vector<Tensor<TensorLabelRank>> &labels, int epochs);


    Layer &operator[](int layer_index);


    template<int OutputRank, int TensorSampleRank>
    Tensor<OutputRank> Predict(const Tensor<TensorSampleRank> &input);


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

protected:
    template<int TensorSampleRank>
    void Forward(const Tensor<TensorSampleRank> &training_sample);

    template<int TensorLabelRank>
    void Backward(const Tensor<TensorLabelRank> &training_label,
                  LossFunction &loss_function);

    void Update(Optimizer &optimizer);


    LossFunction *loss = nullptr;
    Optimizer *opt = nullptr;

};

}

#endif //ORION_SEQUENTIAL_HPP
