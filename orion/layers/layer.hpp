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

    /**
     * Forward propagation for input layers
     *
     * @param inputs   rank 2 tensor training sample
     */
    virtual void Forward(const Tensor<2> &inputs) {};


    /**
     * Backward propagation for output layers
     *
     * @param loss_function   loss object that calculates and records loss
     */
    virtual void Backward(const Loss &loss_function) {};


    /**
     * Forward propagation for hidden layers
     *
     * @param prev   a reference to the previous layer
     */
    virtual void Forward(const Layer &prev) {}


    /**
     * Backward propagation for hidden layers
     *
     * @param next   a reference to the next layer
     */
    virtual void Backward(const Layer &next) {}


    /**
     * Updates the layer's learnable parameters using the
     * provided optimization algorithm
     *
     * @param optimizer   optimizer object that performs layer parameter updates
     */
    virtual void Update(Optimizer &optimizer) = 0;


    /**
     * @return   layer's activation values
     */
    virtual const Tensor<2> &GetOutput2D() const {}
    virtual const Tensor<3> &GetOutput3D() const {}


    /**
     * @return   layer's loss gradients w.r.t. pre-activated output (dL / dZ))
     */
    virtual const Tensor<2> &GetInputGradients2D() const {}


    /**
     * @return   layer's weights
     */
    virtual const Tensor<2> &GetWeights() const {}


    /**
     * @return   layer's loss gradients w.r.t weights (dL / dw)
     */
    virtual const Tensor<2> &GetWeightGradients() const {}


    /**
     * Set the layer's weights
     *
     * @param weights   a rank 2 tensor
     */
    virtual void SetWeights(const Tensor<2> &weights) {}


    /**
     * @return   layer's bias
     */
    virtual const Tensor<2> &GetBias() const {}

    /**
     * Set the layer's bias
     *
     * @param bias   a rank 2 tensor with dimensions (n, 1)
     */
    virtual void SetBias(const Tensor<2> &bias) {}


    /**
     * @return   expected rank of forward propagation's input tensor
     */
    virtual int GetInputRank() const = 0;


    /**
     * @return   expected rank of forward propagation's output tensor
     */
    virtual int GetOutputRank() const = 0;


    // calculate one forward propagation output (e.g. for Dense)
    virtual Tensor<2> operator()(const Tensor<2> &tensor) const {}
    virtual Tensor<2> operator()(const Tensor<3> &tensor) const {}

    std::string name = "layer"; // name of player, to be set by inherited classes
};

}

#endif //ORION_LAYER_HPP
