//
// Created by Macross on 12/14/22.
//

#ifndef ORION_DROPOUT_HPP
#define ORION_DROPOUT_HPP

#include "layer.hpp"

namespace orion
{

template<int InputTensorRank>
class Dropout : public Layer
{
public:
    explicit Dropout(Scalar dropout_rate);

    ~Dropout() override = default;

    void Forward(const Tensor<InputTensorRank> &inputs) override;

    void Forward(const Layer &prev) override;

    void Backward(const Tensor<InputTensorRank> &gradients) override;

    void Backward(const Layer &next) override;

    void Update(Optimizer &) override;

    const Tensor<2> &GetOutput2D() const override;

    const Tensor<3> &GetOutput3D() const override;

    const Tensor<4> &GetOutput4D() const override;

    Tensor<2> GetInputGradients2D() const override;

    Tensor<3> GetInputGradients3D() const override;

    Tensor<4> GetInputGradients4D() const override;

    int GetInputRank() const override;

    int GetOutputRank() const override;


    /**
     * Toggles this layer between inference (dropout rate 0) and
     * training mode (dropout rate as provided in constructor)
     *
     * @param is_training   if true, sets the dropout rate to the constructor provided
     *                      value, else to 0.0 where no features are dropped
     */
    void Training(bool is_training) override;

private:
    Tensor<InputTensorRank> Z;
    Tensor<InputTensorRank> dL_dZ;

    // a tensor of 0s and 1s created from a Bernoulli distribution
    // for zeroing out features when multiplied with the layer input
    Tensor<InputTensorRank> drop_mask;

    Scalar dropout_rate;

    // saves the dropout rate when layer is set for inference mode (no dropout)
    Scalar _dropout_rate_copy;
};

} // namespace orion

#include "dropout.ipp"

#endif //ORION_DROPOUT_HPP
