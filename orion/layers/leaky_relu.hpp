//
// Created by Raymond on 1/19/23.
//

#ifndef ORION_LEAKY_RELU_HPP
#define ORION_LEAKY_RELU_HPP

#include "activation.hpp"

namespace orion
{

template<int TensorRank>
class LeakyReLU : public Activation<Linear, TensorRank>
{
public:
    explicit LeakyReLU(Scalar leak = 0.3);

    void Forward(const Tensor<TensorRank> &inputs) override;

    const Tensor<2> &GetInputGradients2D() override;

    const Tensor<3> &GetInputGradients3D() override;

    const Tensor<4> &GetInputGradients4D() override;

private:
    const Scalar leak = 0.3;

    // multithreading
    Eigen::ThreadPool pool = Eigen::ThreadPool(
            (int) std::thread::hardware_concurrency());
    Eigen::ThreadPoolDevice device = Eigen::ThreadPoolDevice(&pool, 2);
};

} // namespace orion

#include "leaky_relu.ipp"

#endif //ORION_LEAKY_RELU_HPP
