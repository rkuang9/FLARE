//
// Created by macross on 10/13/22.
//

#ifndef ORION_FLATTEN_HPP
#define ORION_FLATTEN_HPP

#include "layer.hpp"

namespace orion
{

/**
 * Flattens the input while preserving the batch dimension
 *
 * @tparam InputTensorRank   input tensor rank (including the batch dimension),
 *                           required to keep track of dimensions to reshape
 *                           gradients to match input
 */
template<int InputTensorRank>
class Flatten : public Layer
{
public:
    Flatten();

    void Forward(const Tensor<InputTensorRank> &input) override;

    void Forward(const Layer &prev) override;

    void Backward(const Tensor<InputTensorRank> &gradients) override;

    void Backward(Layer &next) override;

    void Update(Optimizer &) override;

    const Tensor<2> &GetOutput2D() const override;

    const Tensor<2> &GetInputGradients2D() override;

    const Tensor<3> &GetInputGradients3D() override;

    const Tensor<4> &GetInputGradients4D() override;

    int GetInputRank() const override;

    int GetOutputRank() const override;

private:
    Dims<InputTensorRank> input_dims;

    Tensor<2> Z;
    Tensor<InputTensorRank> dL_dZ;
    Tensor<InputTensorRank> dL_dX;

};

} // namespace orion

#include "flatten.ipp"

#endif //ORION_FLATTEN_HPP