//
// Created by Macross on 1/18/23.
//

#include "activation.hpp"

namespace fl
{

template<typename activation, int TensorRank>
Activation<activation, TensorRank>::Activation():
        pool((int) std::thread::hardware_concurrency()),
        device(&pool, 2)
{
    this->name = "activation";
}


template<typename activation, int TensorRank>
void Activation<activation, TensorRank>::Forward(const Tensor<TensorRank> &inputs)
{
    this->X = inputs;
    this->Z.resize(inputs.dimensions());
    this->Z.template device(this->device) = activation::Activate(inputs);
}


template<typename activation, int TensorRank>
void Activation<activation, TensorRank>::Forward(const Layer &prev)
{
    if constexpr (TensorRank == 2) {
        this->Forward(prev.GetOutput2D());
    }
    else if constexpr (TensorRank == 3) {
        this->Forward(prev.GetOutput3D());
    }
    else if constexpr (TensorRank == 4) {
        this->Forward(prev.GetOutput4D());
    }
    else {
        throw std::logic_error(
                "Activation::Forward UNSUPPORTED INPUT TENSOR RANK");
    }
}


template<typename activation, int TensorRank>
void Activation<activation, TensorRank>::Backward(
        const Tensor<TensorRank> &gradients)
{
    this->dL_dZ = gradients;
}


template<typename activation, int TensorRank>
void Activation<activation, TensorRank>::Backward(Layer &next)
{
    this->dL_dZ.resize(this->Z.dimensions());

    if constexpr (TensorRank == 2) {
        this->dL_dZ = next.GetInputGradients2D();
    }
    else if constexpr (TensorRank == 3) {
        this->dL_dZ = next.GetInputGradients3D();
    }
    else if constexpr (TensorRank == 4) {
        this->dL_dZ = next.GetInputGradients4D();
    }
    else {
        throw std::logic_error(
                "Activation::Backward UNSUPPORTED INPUT TENSOR RANK");
    }
}


template<typename activation, int TensorRank>
void Activation<activation, TensorRank>::Update(Optimizer &)
{
    // nothing to do
}


template<typename activation, int TensorRank>
const Tensor<2> &Activation<activation, TensorRank>::GetOutput2D() const
{
    if constexpr (TensorRank != 2) {
        throw std::logic_error(
                "Activation::GetOutput2D CALLED ON A RANK " +
                std::to_string(TensorRank) + " TENSOR");
    }

    return this->Z;
}


template<typename activation, int TensorRank>
const Tensor<3> &Activation<activation, TensorRank>::GetOutput3D() const
{
    if constexpr (TensorRank != 3) {
        throw std::logic_error(
                "Activation::GetOutput3D CALLED ON A RANK " +
                std::to_string(TensorRank) + " TENSOR");
    }

    return this->Z;
}


template<typename activation, int TensorRank>
const Tensor<4> &Activation<activation, TensorRank>::GetOutput4D() const
{
    if constexpr (TensorRank != 4) {
        throw std::logic_error(
                "Activation::GetOutput4D CALLED ON A RANK " +
                std::to_string(TensorRank) + " TENSOR");
    }

    return this->Z;
}


template<typename activation, int TensorRank>
const Tensor<2> &Activation<activation, TensorRank>::GetInputGradients2D()
{
    if constexpr (TensorRank != 2) {
        throw std::logic_error(
                "Activation::GetInputGradients2D CALLED ON A RANK " +
                std::to_string(TensorRank) + " TENSOR");
    }

    // forward pass: z = g(x)
    // backward pass (input): dL/dX = dL/dZ * dZ/dX = dL/dZ * g'(x)
    this->dL_dX.resize(this->X.dimensions());
    this->dL_dX.template device(this->device) =
            this->dL_dZ * activation::Gradients(this->X);
    return this->dL_dX;
}


template<typename activation, int TensorRank>
const Tensor<3> &Activation<activation, TensorRank>::GetInputGradients3D()
{
    if constexpr (TensorRank != 3) {
        throw std::logic_error(
                "Activation::GetInputGradients3D CALLED ON A RANK " +
                std::to_string(TensorRank) + " TENSOR");
    }

    this->dL_dX.resize(this->X.dimensions());
    this->dL_dX.template device(this->device) =
            this->dL_dZ * activation::Gradients(this->X);
    return this->dL_dX;
}


template<typename activation, int TensorRank>
const Tensor<4> &Activation<activation, TensorRank>::GetInputGradients4D()
{
    if constexpr (TensorRank != 4) {
        throw std::logic_error(
                "Activation::GetInputGradients4D CALLED ON A RANK " +
                std::to_string(TensorRank) + " TENSOR");
    }

    this->dL_dX.resize(this->X.dimensions());
    this->dL_dX.template device(this->device) =
            this->dL_dZ * activation::Gradients(this->X);
    return this->dL_dX;
}


template<typename activation, int TensorRank>
int Activation<activation, TensorRank>::GetInputRank() const
{
    return TensorRank;
}


template<typename activation, int TensorRank>
int Activation<activation, TensorRank>::GetOutputRank() const
{
    return TensorRank;
}

} // namespace fl