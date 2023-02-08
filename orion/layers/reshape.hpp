//
// Created by Raymond on 1/16/23.
//

#ifndef ORION_RESHAPE_H
#define ORION_RESHAPE_H

#include "layer.hpp"

namespace orion
{

template<int InputTensorRank, int OutputTensorRank>
class Reshape : public Layer
{
public:
    explicit Reshape(const Dims<OutputTensorRank> &output_dims);

    explicit Reshape(const std::vector<Eigen::Index> &output_dims);

    void Forward(const Tensor<InputTensorRank> &inputs) override;

    void Forward(const Layer &prev) override;

    void Backward(const Tensor<OutputTensorRank> &gradients) override;

    void Backward(Layer &next) override;

    void Update(Optimizer &) override;

    const Tensor<2> &GetOutput2D() const override;

    const Tensor<3> &GetOutput3D() const override;

    const Tensor<4> &GetOutput4D() const override;

    const Tensor<2> &GetInputGradients2D() override;

    const Tensor<3> &GetInputGradients3D() override;

    const Tensor<4> &GetInputGradients4D() override;

    int GetInputRank() const override;

    int GetOutputRank() const override;

private:
    Dims<OutputTensorRank> output_dims;
    Dims<InputTensorRank> input_dims;
    Eigen::Index unfixed_dim = -1; // e.g. can be 0 since batch dim may vary

    Tensor<OutputTensorRank> Z;
    Tensor<OutputTensorRank> dL_dZ;
    Tensor<InputTensorRank> dL_dX;

    // multithreading
    Eigen::ThreadPool pool;
    Eigen::ThreadPoolDevice device;
};

} // namespace orion

#include "reshape.ipp"

#endif //ORION_RESHAPE_H
