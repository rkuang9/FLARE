//
// Created by Macross on 1/19/23.
//

#include "leaky_relu.hpp"

namespace orion
{

template<int TensorRank>
LeakyReLU<TensorRank>::LeakyReLU(Scalar leak):
        leak(leak),
        pool((int) std::thread::hardware_concurrency()),
        device(&pool, 2)
{
    this->name = "leakyrelu";
}


template<int TensorRank>
void LeakyReLU<TensorRank>::Forward(const Tensor<TensorRank> &inputs)
{
    this->X = inputs;
    auto zero = static_cast<Scalar>(0.0);
    this->Z = (inputs >= zero).select(inputs, inputs * this->leak);
}


template<int TensorRank>
const Tensor<2> &LeakyReLU<TensorRank>::GetInputGradients2D()
{
    if constexpr (TensorRank != 2) {
        throw std::logic_error(
                "LeakyReLU::GetInputGradients2D CALLED ON A RANK " +
                std::to_string(TensorRank) + " TENSOR");
    }

    auto one = static_cast<Scalar>(1.0);
    this->dL_dX.resize(this->X.dimensions());
    this->dL_dX.template device(this->device) =
            this->dL_dZ * (this->X >= this->leak)
                    .select(this->X.constant(one), this->X.constant(this->leak));
    return this->dL_dX;
}


template<int TensorRank>
const Tensor<3> &LeakyReLU<TensorRank>::GetInputGradients3D()
{
    if constexpr (TensorRank != 3) {
        throw std::logic_error(
                "LeakyReLU::GetInputGradients3D CALLED ON A RANK " +
                std::to_string(TensorRank) + " TENSOR");
    }

    auto one = static_cast<Scalar>(1.0);
    this->dL_dX.resize(this->X.dimensions());
    this->dL_dX.template device(this->device) =
            this->dL_dZ * (this->X >= this->leak)
                    .select(this->X.constant(one), this->X.constant(this->leak));
    return this->dL_dX;
}


template<int TensorRank>
const Tensor<4> &LeakyReLU<TensorRank>::GetInputGradients4D()
{
    if constexpr (TensorRank != 4) {
        throw std::logic_error(
                "LeakyReLU::GetInputGradients4D CALLED ON A RANK " +
                std::to_string(TensorRank) + " TENSOR");
    }
    auto one = static_cast<Scalar>(1.0);
    this->dL_dX.resize(this->X.dimensions());
    this->dL_dX.template device(this->device) =
            this->dL_dZ * (this->X >= this->leak)
                    .select(this->X.constant(one), this->X.constant(this->leak));
    return this->dL_dX;
}

} // namespace orion