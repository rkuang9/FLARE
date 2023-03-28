//
// Created by macross on 10/13/22.
//

#include "flatten.hpp"

namespace fl
{

template<int InputTensorRank>
Flatten<InputTensorRank>::Flatten()
{
    this->name = "flatten";
}


template<int InputTensorRank>
void Flatten<InputTensorRank>::Forward(const Tensor<InputTensorRank> &input)
{
    this->input_dims = input.dimensions();

    this->Z.resize(Dims<2>(this->input_dims.front(),
                    this->input_dims.TotalSize() / this->input_dims.front()));
    this->Z.device(this->device) = input.reshape(
            Dims<2>(this->input_dims.front(),
                    this->input_dims.TotalSize() / this->input_dims.front()));
}


template<int InputTensorRank>
void Flatten<InputTensorRank>::Forward(const Layer &prev)
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
        throw std::invalid_argument(
                "Flatten currently only accepts Tensor<3> and Tensor<4>");
    }
}


template<int InputTensorRank>
void Flatten<InputTensorRank>::Backward(const Tensor<InputTensorRank> &gradients)
{
    fl_assert(this->Z.dimensions() == gradients.dimensions(),
              this->name << "::Backward expected gradient dimension "
                         << this->Z.dimensions() << ", instead got "
                         << gradients.dimensions());

    this->dL_dZ.resize(gradients.dimensions());

    if constexpr (InputTensorRank == 2) {
        this->dL_dZ.device(this->device) = gradients;
    }
    else {
        this->dL_dZ.device(this->device) = gradients.reshape(this->input_dims);
    }
}


template<int InputTensorRank>
void Flatten<InputTensorRank>::Backward(Layer &next)
{
    this->dL_dZ = next.GetInputGradients2D().reshape(this->input_dims);
}


template<int InputTensorRank>
void Flatten<InputTensorRank>::Update(Optimizer &)
{
    // nothing to do, avoids calling Layer::Update
}


template<int InputTensorRank>
const Tensor<2> &Flatten<InputTensorRank>::GetOutput2D() const
{
    return this->Z;
}


#ifdef _WIN32
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
#elif defined __unix__ || defined __APPLE__ || defined __linux__
#pragma GCC diagnostic ignored "-Wreturn-stack-address"
#endif


template<int InputTensorRank>
const Tensor<2> &Flatten<InputTensorRank>::GetInputGradients2D()
{
    if constexpr (InputTensorRank != 2) {
        std::ostringstream error;
        error << "Flatten::GetInputGradients2D called on a " <<
              InputTensorRank << "D tensor";
        throw std::logic_error(error.str());
    }

    return this->dL_dZ;
}


#ifdef _WIN32
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
#elif defined __unix__ || defined __APPLE__ || defined __linux__
#pragma GCC diagnostic ignored "-Wreturn-stack-address"
#endif


template<int InputTensorRank>
const Tensor<3> &Flatten<InputTensorRank>::GetInputGradients3D()
{
    if constexpr (InputTensorRank != 3) {
        std::ostringstream error;
        error << "Flatten::GetInputGradients3D called on a " <<
              InputTensorRank << "D tensor";
        throw std::logic_error(error.str());
    }

    return this->dL_dZ;
}


#ifdef _WIN32
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
#elif defined __unix__ || defined __APPLE__ || defined __linux__
#pragma GCC diagnostic ignored "-Wreturn-stack-address"
#endif


template<int InputTensorRank>
const Tensor<4> &Flatten<InputTensorRank>::GetInputGradients4D()
{
    if constexpr (InputTensorRank != 4) {
        std::ostringstream error;
        error << "Flatten::GetInputGradients4D called on a " <<
              InputTensorRank << "D tensor";
        throw std::logic_error(error.str());
    }

    return this->dL_dZ;
}


template<int InputTensorRank>
int Flatten<InputTensorRank>::GetInputRank() const
{
    return InputTensorRank;
}


template<int InputTensorRank>
int Flatten<InputTensorRank>::GetOutputRank() const
{
    return 2;
}

} // namespace fl