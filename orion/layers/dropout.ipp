//
// Created by Macross on 12/14/22.
//

#include "dropout.hpp"

namespace orion
{

template<int InputTensorRank>
Dropout<InputTensorRank>::Dropout(Scalar dropout_rate)
        : dropout_rate(dropout_rate),
          _dropout_rate_copy(dropout_rate)
{
    if (dropout_rate < 0 || dropout_rate >= 1) {
        throw std::invalid_argument("Dropout RATE MUST BE WITHIN RANGE [0,1)");
    }

    this->name = "dropout";
}


template<int InputTensorRank>
void Dropout<InputTensorRank>::Forward(const Tensor<InputTensorRank> &inputs)
{
    // generate a tensor on a Bernoulli distribution where
    // (1 - dropout_chance) is the chance of keeping an input feature
    this->drop_mask = RandomBernoulli(inputs.dimensions(),
                                      1.0 - this->dropout_rate);
    this->Z.resize(inputs.dimensions());
    this->Z.template device(this->device) =
            inputs * this->drop_mask / static_cast<Scalar>(1 - this->dropout_rate);
}


template<int InputTensorRank>
void Dropout<InputTensorRank>::Forward(const Layer &prev)
{
    if constexpr (InputTensorRank == 2) {
        this->Forward(prev.GetOutput2D());
    }
    else if constexpr (InputTensorRank == 3) {
        this->Forward(prev.GetOutput3D());
    }
    else if constexpr (InputTensorRank == 4) {
        this->Forward(prev.GetOutput4D());
    }
    else {
        throw std::logic_error("Dropout::Backward unsupported rank");
    }
}


template<int InputTensorRank>
void Dropout<InputTensorRank>::Backward(const Tensor<InputTensorRank> &gradients)
{
    this->dL_dZ = gradients;
}


template<int InputTensorRank>
void Dropout<InputTensorRank>::Backward(Layer &next)
{
    if constexpr (InputTensorRank == 2) {
        this->dL_dZ = next.GetInputGradients2D();
    }
    else if constexpr (InputTensorRank == 3) {
        this->dL_dZ = next.GetInputGradients3D();
    }
    else if constexpr (InputTensorRank == 4) {
        this->dL_dZ = next.GetInputGradients4D();
    }
    else {
        throw std::logic_error("Dropout::Backward INPUT TENSOR RANK");
    }
}


template<int InputTensorRank>
void Dropout<InputTensorRank>::Update(Optimizer &)
{
    // nothing to do, avoids calling Layer::Update
}


#ifdef _WIN32
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
#elif defined __unix__ || defined __APPLE__ || defined __linux__
#pragma GCC diagnostic ignored "-Wreturn-stack-address"
#endif


template<int InputTensorRank>
const Tensor<2> &Dropout<InputTensorRank>::GetOutput2D() const
{
    return this->Z;
}


#ifdef _WIN32
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
#elif defined __unix__ || defined __APPLE__ || defined __linux__
#pragma GCC diagnostic ignored "-Wreturn-stack-address"
#endif


template<int InputTensorRank>
const Tensor<3> &Dropout<InputTensorRank>::GetOutput3D() const
{
    return this->Z;
}


#ifdef _WIN32
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
#elif defined __unix__ || defined __APPLE__ || defined __linux__
#pragma GCC diagnostic ignored "-Wreturn-stack-address"
#endif


template<int InputTensorRank>
const Tensor<4> &Dropout<InputTensorRank>::GetOutput4D() const
{
    return this->Z;
}


#ifdef _WIN32
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
#elif defined __unix__ || defined __APPLE__ || defined __linux__
#pragma GCC diagnostic ignored "-Wreturn-stack-address"
#endif


template<int InputTensorRank>
const Tensor<2> &Dropout<InputTensorRank>::GetInputGradients2D()
{
    if constexpr (InputTensorRank != 2) {
        throw std::logic_error(
                "Dropout::GetInputGradients2D CALLED ON A RANK " +
                std::to_string(InputTensorRank) + " TENSOR");
    }

    this->dL_dX.resize(this->Z.dimensions());
    this->dL_dX.template device(this->device) =
            this->dL_dZ * this->drop_mask /
            static_cast<Scalar>(1.0 - this->dropout_rate);
    return this->dL_dX;
}


#ifdef _WIN32
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
#elif defined __unix__ || defined __APPLE__ || defined __linux__
#pragma GCC diagnostic ignored "-Wreturn-stack-address"
#endif


template<int InputTensorRank>
const Tensor<3> &Dropout<InputTensorRank>::GetInputGradients3D()
{
    if constexpr (InputTensorRank != 3) {
        throw std::logic_error(
                "Dropout::GetInputGradients3D CALLED ON A RANK " +
                std::to_string(InputTensorRank) + " TENSOR");
    }

    this->dL_dX.resize(this->Z.dimensions());
    this->dL_dX.template device(this->device) =
            this->dL_dZ * this->drop_mask /
            static_cast<Scalar>(1.0 - this->dropout_rate);
    return this->dL_dX;
}


#ifdef _WIN32
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
#elif defined __unix__ || defined __APPLE__ || defined __linux__
#pragma GCC diagnostic ignored "-Wreturn-stack-address"
#endif


template<int InputTensorRank>
const Tensor<4> &Dropout<InputTensorRank>::GetInputGradients4D()
{
    if constexpr (InputTensorRank != 4) {
        throw std::logic_error(
                "Dropout::GetInputGradients4D CALLED ON A RANK " +
                std::to_string(InputTensorRank) + " TENSOR");
    }

    this->dL_dX.resize(this->Z.dimensions());
    this->dL_dX.template device(this->device) =
            this->dL_dZ * this->drop_mask /
            static_cast<Scalar>(1.0 - this->dropout_rate);
    return this->dL_dX;
}


template<int InputTensorRank>
int Dropout<InputTensorRank>::GetInputRank() const
{
    return InputTensorRank;
}


template<int InputTensorRank>
int Dropout<InputTensorRank>::GetOutputRank() const
{
    return InputTensorRank;
}


template<int InputTensorRank>
void Dropout<InputTensorRank>::Training(bool is_training)
{
    // no dropout during inference, everything is kept
    this->dropout_rate = is_training ? _dropout_rate_copy : 0.0;
}

} // namespace orion