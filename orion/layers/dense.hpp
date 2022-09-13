//
// Created by macross on 8/7/22.
//

#ifndef ORION_DENSE_HPP
#define ORION_DENSE_HPP

#include "layer.hpp"

namespace orion
{

template<typename Activation = Linear>
class Dense : public Layer
{
public:
    explicit Dense(int inputs, int outputs, bool use_bias,
                   const Initializer &initializer = GlorotUniform());

    ~Dense() override = default;

    /**
     * Forward propagation, computes "z" and "a" with the formulas
     * z = wx + b, a = g(z) where g is the chose Activation template type
     *
     * @param tensor   a rank 2 tensor training sample
     */
    void Forward(const Tensor<2> &input) override;


    /**
     * Forward propagation for hidden layers, passes the previous layer's
     * output as input for Forward(input)
     *
     * @param prev   a reference to the previous layer
     */
    void Forward(const Layer &prev) override;

    /**
     * Backward propagation for hidden layers, passes next layer's
     * weights and gradients to Backward(gradients)
     *
     * @param next   a reference to the next layer
     */
    void Backward(const Layer &next) override;


    /**
     * Backward propagation for the output layer, passes the loss gradients
     * w.r.t. output layer's output to Backward(gradients)
     *
     * @param loss_function   a reference to the loss object
     */
    void Backward(const Loss &loss_function) override;


    /**
     * Updates the layer's weights and bias with dL/dw, dL/db according
     * to the optimizer algorithm
     *
     * @param optimizer   a reference to the optimizer object
     */
    void Update(Optimizer &optimizer) override;


    // getter setters

    /**
     * @return   layer's activation values
     */
    const Tensor<2> &GetOutput2D() const override;


    /**
     * @return   layer's loss gradients w.r.t. pre-activated output (dL / dZ))
     */
    const Tensor<2> &GetInputGradients2D() const override;


    /**
     * @return   layer's weights
     */
    const Tensor<2> &GetWeights() const override;


    /**
     * @return   loss gradients w.r.t. weights (dL / dw)
     */
    const Tensor<2> &GetWeightGradients() const override;


    /**
     * Set layer's weights
     *
     * @param weights   custom weights with dimensions [output units, input units]
     */
    void SetWeights(const Tensor<2> &weights) override;

    /**
     * @return   layer's bias
     */
    const Tensor<2> &GetBias() const override;


    /**
     * Set layer's bias
     *
     * @param bias   custom bias with dimensions [output units, 1]
     */
    void SetBias(const Tensor<2> &bias) override;


    /**
     * @return   expected rank of forward propagation's input tensor
     */
    int GetInputRank() const override;


    /**
     * @return   expected rank of forward propagation's output tensor
     */
    int GetOutputRank() const override;


    /**
     * Independently compute one forward pass
     *
     * @param tensor   a rank 2 tensor training sample
     * @return   activated units as a result of the forward pass
     */
    Tensor<2> operator()(const Tensor<2> &tensor) const override;

private:
    /**
     * Backward propagation, computes the loss gradients w.r.t. Z (dL / dZ)
     * and the loss gradients w.r.t. weights and bias (dL / dw, dL / db)
     *
     * @param gradients   loss gradients, source will differ based on calling layer
     */
    void Backward(const Tensor2D &gradients);

    bool use_bias = true;

    Tensor<2> X; // layer input matrix, stacked column-wise
    Tensor<2> Z; // weighted input matrix, Z = w*X + b
    Tensor<2> A; // activated input matrix after applying g(Z)
    Tensor<2> dL_dZ; // derivative of loss w.r.t. inputs, same dim as X, Z, and A

    Tensor<2> w; // weights matrix
    Tensor<2> b; // bias vector
    Tensor<2> dL_dw; // loss gradients w.r.t. weights
    Tensor<2> dL_db; // loss gradients w.r.t. bias
};

} // namespace orion

#include "dense.ipp"

#endif //ORION_DENSE_HPP
