//
// Created by macross on 8/7/22.
//

#ifndef FLARE_LAYER_HPP
#define FLARE_LAYER_HPP

#include "flare/fl_types.hpp"
#include "flare/fl_assert.hpp"
#include "flare/weights/weights.hpp"
#include "flare/weights/glorot.hpp"
#include "flare/optimizers/optimizer.hpp"
#include <fstream>

namespace fl
{

class Layer
{
public:
    Layer() = default;

    Layer(int input_rank, int output_rank) :
            input_rank(input_rank),
            output_rank(output_rank)
    {
    }


    virtual ~Layer() = default;


    /**
     * Forward propagation for input layers
     *
     * @param inputs   rank 2 tensor training sample
     */
    virtual void Forward(const Tensor<2> &inputs)
    {
        throw std::logic_error(
                "An error occurred, base class Layer Forward was called on " +
                this->name);
    }


    /**
     * Forward propagation for input layers
     *
     * @param inputs   rank 3 tensor training sample
     */
    virtual void Forward(const Tensor<3> &inputs)
    {
        throw std::logic_error(
                "An error occurred, base class Layer Forward was called on " +
                this->name);
    }


    /**
     * Forward propagation for input layers
     *
     * @param inputs   rank 3 tensor training sample
     */
    virtual void Forward(const Tensor<4> &inputs)
    {
        throw std::logic_error(
                "An error occurred, base class Layer Forward was called on " +
                this->name);
    }


    /**
     * Forward propagation for hidden layers
     *
     * @param prev   a reference to the previous layer
     */
    virtual void Forward(const Layer &prev)
    {
        throw std::logic_error(
                "An error occurred, base class Layer Forward was called on " +
                this->name);
    }


    virtual void Backward(const Tensor<2> &gradients)
    {
        throw std::logic_error(
                "An error occurred, base class Layer Backward was called on " +
                this->name);
    }


    virtual void Backward(const Tensor<3> &gradients)
    {
        throw std::logic_error(
                "An error occurred, base class Layer Backward was called on " +
                this->name);
    }


    virtual void Backward(const Tensor<4> &gradients)
    {
        throw std::logic_error(
                "An error occurred, base class Layer Backward was called on " +
                this->name);
    }


    /**
     * Backward propagation for hidden layers
     *
     * @param next   a reference to the next layer
     */
    virtual void Backward(Layer &next)
    {
        throw std::logic_error(
                "An error occurred, base class Layer Backward was called on " +
                this->name);
    }


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
    virtual const Tensor<2> &GetOutput2D() const
    {
        throw std::logic_error(
                "An error occurred, base class Layer GetOutput2D was called on " +
                this->name);
    }


    virtual const Tensor<3> &GetOutput3D() const
    {
        throw std::logic_error(
                "An error occurred, base class Layer GetOutput3D was called on " +
                this->name);
    }


    virtual const Tensor<4> &GetOutput4D() const
    {
        throw std::logic_error(
                "An error occurred, base class Layer GetOutput4D was called on " +
                this->name);
    }


    /**
     * Input gradients to be fed into previous layer. This is calculated as needed
     * so results will not be as expected is called after te layer updated
     * @return   layer's loss gradients w.r.t. pre-activated output (dL / dZ))
     */
    virtual const Tensor<2> &GetInputGradients2D()
    {
        throw std::logic_error(
                "An error occurred, base class Layer GetInputGradients2D was called on " +
                this->name);
    }


    // same documentation as GetInputGradients2D()
    virtual const Tensor<3> &GetInputGradients3D()
    {
        throw std::logic_error(
                "An error occurred, base class Layer GetInputGradients3D was called on " +
                this->name);
    }


    // same documentation as GetInputGradients2D()
    virtual const Tensor<4> &GetInputGradients4D()
    {
        throw std::logic_error(
                "An error occurred, base class Layer GetInputGradients4D was called on " +
                this->name);
    }


    virtual std::vector<Tensor<2>> GetWeights2D() const
    {
        return {};
    }


    virtual std::vector<Tensor<3>> GetWeights3D() const
    {
        return {};
    }


    virtual std::vector<Tensor<4>> GetWeights4D() const
    {
        return {};
    }


    /**
     * @return   layer's loss gradients w.r.t weights (dL / dw)
     */
    virtual std::vector<Tensor<2>> GetWeightGradients2D() const
    {
        throw std::logic_error(
                "An error occurred, base class Layer GetWeightGradients was called  on " +
                this->name);
    }


    virtual std::vector<Tensor<3>> GetWeightGradients3D() const
    {
        throw std::logic_error(
                "An error occurred, base class Layer GetWeightGradients3D was called  on " +
                this->name);
    }


    virtual std::vector<Tensor<4>> GetWeightGradients4D() const
    {
        throw std::logic_error(
                "An error occurred, base class Layer GetWeightGradients4D was call on " +
                this->name);
    }


    /**
     * Set the layer's weights
     *
     * @param weights   a rank 2 tensor
     */
    virtual void SetWeights(const std::vector<Tensor<2>> &weights)
    {
        throw std::logic_error(
                "An error occurred, base class Layer SetWeights(<2>) was called on " +
                this->name);
    }


    virtual void SetWeights(const std::vector<Tensor<3>> &weights)
    {
        throw std::logic_error(
                "An error occurred, base class Layer SetWeights(<3>) was called on " +
                this->name);
    }


    virtual void SetWeights(const std::vector<Tensor<4>> &weights)
    {
        throw std::logic_error(
                "An error occurred, base class Layer SetWeights(<4>) was called on " +
                this->name);
    }


    /**
     * @return   expected rank of forward propagation's input tensor
     */
    virtual int GetInputRank() const
    {
        return this->input_rank;
    };


    /**
     * @return   expected rank of forward propagation's output tensor
     */
    virtual int GetOutputRank() const
    {
        return this->output_rank;
    }


    virtual void Training(bool is_training)
    {}


    virtual void Save(const std::string &path)
    {}


    virtual void Load(const std::string &path)
    {}


    std::string name = "layer"; // name of layer, to be set by inherited classes
    int input_rank = -1;
    int output_rank = -1;
};

}

#endif //FLARE_LAYER_HPP
