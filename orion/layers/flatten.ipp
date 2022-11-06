//
// Created by macross on 10/13/22.
//

#include "flatten.hpp"

namespace orion
{


template<int InputTensorRank>
void Flatten<InputTensorRank>::Forward(const Tensor<InputTensorRank> &input)
{
    this->input_dims = input.dimensions();

    if constexpr (InputTensorRank == 2) {
        // why someone would flatten a fully-connected layer, idk but here it is
        this->Z = input;
    }
    else {
        // reshape input from (N,X,Y,Z) where N=batch_size, to (X*Y*Z, N)
        this->Z = input.reshape(
                Dims<2>(this->input_dims.TotalSize() / this->input_dims.front(),
                        this->input_dims.front()));
    }
}


template<int InputTensorRank>
void Flatten<InputTensorRank>::Forward(const Layer &prev)
{
    if constexpr (InputTensorRank == 4) {
        this->Forward(prev.GetOutput4D());
    }
    else if constexpr (InputTensorRank == 3) {
        this->Forward(prev.GetOutput3D());
    }
    else {
        throw std::invalid_argument(
                "Flatten currently only accepts Tensor<3> and Tensor<4>");
    }
}


template<int InputTensorRank>
void Flatten<InputTensorRank>::Backward(const LossFunction &loss_function)
{
    if constexpr (InputTensorRank == 2) {
        // divide by number of output units in a single batch, [output units, batch]
        this->dL_dZ = loss_function.GetGradients2D() /
                      static_cast<Scalar>(this->input_dims[1]);
    }
    else {
        // divide by number of output units in a single batch, [batch, etc...]
        this->dL_dZ = loss_function.GetGradients2D().reshape(this->input_dims) /
                      static_cast<Scalar>(this->input_dims.TotalSize() /
                                          this->input_dims[0]);
    }
}


template<int InputTensorRank>
void Flatten<InputTensorRank>::Backward(const Layer &next)
{
    // undo flatten by undoing and reshaping to input dims
    if constexpr (InputTensorRank == 2) {
        this->dL_dZ = next.GetInputGradients2D();
    }
    else {
        this->dL_dZ = next.GetInputGradients2D().reshape(this->input_dims);
    }
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


#pragma GCC diagnostic ignored "-Wreturn-local-addr"

template<int InputTensorRank>
const Tensor<2> &Flatten<InputTensorRank>::GetInputGradients2D() const
{
    if constexpr (InputTensorRank != 2) {
        std::ostringstream error;
        error << "Flatten::GetInputGradients2D called on a " <<
              InputTensorRank << "D tensor";
        throw std::logic_error(error.str());
    }

    return this->dL_dZ;
}


#pragma GCC diagnostic ignored "-Wreturn-local-addr"

template<int InputTensorRank>
const Tensor<3> &Flatten<InputTensorRank>::GetInputGradients3D() const
{
    if constexpr (InputTensorRank != 3) {
        std::ostringstream error;
        error << "Flatten::GetInputGradients3D called on a " <<
              InputTensorRank << "D tensor";
        throw std::logic_error(error.str());
    }

    return this->dL_dZ;
}


#pragma GCC diagnostic ignored "-Wreturn-local-addr"

template<int InputTensorRank>
const Tensor<4> &Flatten<InputTensorRank>::GetInputGradients4D() const
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

} // namespace orion