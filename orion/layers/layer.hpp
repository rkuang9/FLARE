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
    virtual void Forward(const Tensor<2> &inputs)
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    /**
     * Forward propagation for input layers
     *
     * @param inputs   rank 3 tensor training sample
     */
    virtual void Forward(const Tensor<3> &inputs)
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    /**
     * Forward propagation for input layers
     *
     * @param inputs   rank 3 tensor training sample
     */
    virtual void Forward(const Tensor<4> &inputs)
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    /**
     * Forward propagation for hidden layers
     *
     * @param prev   a reference to the previous layer
     */
    virtual void Forward(const Layer &prev)
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    /**
     * Backward propagation for output layers
     *
     * @param loss_function   loss object that calculates and records loss
     */
    virtual void Backward(const LossFunction &loss_function)
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    /**
     * Backward propagation for hidden layers
     *
     * @param next   a reference to the next layer
     */
    virtual void Backward(const Layer &next)
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    /**
     * Updates the layer's learnable parameters using the
     * provided optimization algorithm
     *
     * @param optimizer   optimizer object that performs layer parameter updates
     */
    virtual void Update(Optimizer &optimizer)
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    /**
     * @return   layer's activation values
     */
    virtual const Tensor<2> &GetOutput2D() const
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    virtual const Tensor<3> &GetOutput3D() const
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    virtual const Tensor<4> &GetOutput4D() const
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    /**
     * @return   layer's loss gradients w.r.t. pre-activated output (dL / dZ))
     */
    virtual const Tensor<2> &GetInputGradients2D() const
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    virtual const Tensor<3> &GetInputGradients3D() const
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    virtual const Tensor<4> &GetInputGradients4D() const
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    /**
     * @return   layer's weights
     */
    virtual const Tensor<2> &GetWeights() const
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    virtual const Tensor<4> &GetWeights4D() const
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    /**
     * @return   layer's loss gradients w.r.t weights (dL / dw)
     */
    virtual const Tensor<2> &GetWeightGradients() const
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    virtual const Tensor<4> &GetWeightGradients4D() const
    {}


    /**
     * Set the layer's weights
     *
     * @param weights   a rank 2/3 tensor
     */
    virtual void SetWeights(const Tensor<2> &weights)
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    virtual void SetWeights(const Tensor<4> &weights)
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    /**
     * @return   layer's bias
     */
    virtual const Tensor<2> &GetBias() const
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    /**
     * Set the layer's bias
     *
     * @param bias   a rank 2 tensor with dimensions (n, 1)
     */
    virtual void SetBias(const Tensor<2> &bias)
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    virtual void SetBias(const Tensor<4> &bias)
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    /**
     * @return   expected rank of forward propagation's input tensor
     */
    virtual int GetInputRank() const = 0;


    /**
     * @return   expected rank of forward propagation's output tensor
     */
    virtual int GetOutputRank() const = 0;


    // calculate one forward propagation output (e.g. for Dense)
    virtual Tensor<2> operator()(const Tensor<2> &tensor) const
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    virtual Tensor<2> operator()(const Tensor<3> &tensor) const
    {
        throw std::logic_error("An error occurred, base class Layer was called");
    }


    std::string name = "layer"; // name of player, to be set by inherited classes
};

}

#endif //ORION_LAYER_HPP
