//
// Created by macross on 8/7/22.
//

#ifndef ORION_LAYER_HPP
#define ORION_LAYER_HPP

#include "orion/orion_types.hpp"
#include "orion/orion_assert.hpp"
#include "orion/activations/include_activations.hpp"
#include "orion/weights/include_weights.hpp"
#include "orion/optimizers/include_optimizers.hpp"
#include "orion/loss/include_loss.hpp"

namespace orion
{

class Layer
{
public:
    virtual ~Layer() = default;

    // for input layer
    virtual void Forward(const Tensor<2> &inputs) {};

    // output layer back propagation
    virtual void Backward(const Loss &loss_function) {};

    // hidden layer forward propagation
    virtual void Forward(const Layer &prev) {}

    // hidden layer back propagation
    virtual void Backward(const Layer &next) {}

    // update weights using an optimizer
    virtual void Update(Optimizer &optimizer) = 0;


    // getter, setters
    virtual const Tensor<2> &GetOutput2D() const {}
    virtual const Tensor<3> &GetOutput3D() const {}

    virtual const Tensor<2> &GetInputGradients2D() const {}

    virtual Tensor<2> GetGradients() const = 0; // dL/dW term

    virtual const Tensor<2> &GetWeights() const {}
    virtual const Tensor<2> &GetWeightGradients() const {}

    virtual void SetWeights(const Tensor<2> &weights) {}

    virtual Tensor<2> &Bias() {}

    virtual int GetInputRank() const = 0;

    virtual int GetOutputRank() const = 0;




    // calculate one forward propagation output (e.g. for Dense)
    virtual Tensor<2> operator()(const Tensor<2> &tensor) const {}
    virtual Tensor<2> operator()(const Tensor<3> &tensor) const {}

    std::string name = "layer";
protected:
    ContractDim matmul = {Axes(1, 0)}; // left's row * right's col

};

}

#endif //ORION_LAYER_HPP
