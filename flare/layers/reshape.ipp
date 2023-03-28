//
// Created by Raymond on 1/16/23.
//

#include "reshape.hpp"

namespace fl
{


template<int InputTensorRank, int OutputTensorRank>
Reshape<InputTensorRank, OutputTensorRank>::Reshape(
        const Dims <OutputTensorRank> &output_dims) :
        output_dims(output_dims),
        pool((int) std::thread::hardware_concurrency()),
        device(&pool, 2)
{
    for (int i = 0; i < output_dims.size(); i++) {
        if (output_dims[i] == -1) {
            if (this->unfixed_dim != -1) {
                throw std::invalid_argument(
                        "Reshape THERE MAY ONLY BE ONE UNFIXED DIM (-1) IN OUTPUT DIMS");
            }

            this->unfixed_dim = i;
        }
    }

    this->name = "reshape";
}


template<int InputTensorRank, int OutputTensorRank>
Reshape<InputTensorRank, OutputTensorRank>::Reshape(
        const std::vector<Eigen::Index> &output_dims) :
        pool((int) std::thread::hardware_concurrency()),
        device(&pool, 2)
{
    if (output_dims.size() == OutputTensorRank) {
        std::copy_n(output_dims.begin(), OutputTensorRank,
                    this->output_dims.begin());
    }
    else {
        throw std::invalid_argument(
                "Reshape EXPECTED " + std::to_string(OutputTensorRank) +
                " OUTPUT DIMS, GOT" + std::to_string(output_dims.size()));
    }

    for (int i = 0; i < output_dims.size(); i++) {
        if (output_dims[i] == -1) {
            if (this->unfixed_dim != -1) {
                throw std::invalid_argument(
                        "Reshape THERE MAY ONLY BE ONE UNFIXED DIM (-1) IN OUTPUT DIMS");
            }

            this->unfixed_dim = i;
        }
    }

    this->name = "reshape";
}


template<int InputTensorRank, int OutputTensorRank>
void Reshape<InputTensorRank, OutputTensorRank>::Forward(
        const Tensor <InputTensorRank> &inputs)
{
    Dims<OutputTensorRank> reshape = this->output_dims;
    this->input_dims = inputs.dimensions();

    if (this->unfixed_dim != -1) {
        reshape[this->unfixed_dim] = inputs.size() /
                                     std::abs(this->output_dims.TotalSize());
    }

    if (reshape.TotalSize() != inputs.size()) {
        std::ostringstream error_msg;
        error_msg << "Reshape::Forward SIZES ARE NO EQUAL, TRIED RESHAPING " <<
                  inputs.dimensions() << " TO " << reshape;
        throw std::logic_error(error_msg.str());
    }

    this->Z.resize(reshape);
    this->Z.template device(this->device) = inputs.reshape(reshape);
}


template<int InputTensorRank, int OutputTensorRank>
void Reshape<InputTensorRank, OutputTensorRank>::Forward(const Layer &prev)
{
    fl_assert(InputTensorRank >= 2 && InputTensorRank <= 4 &&
              OutputTensorRank >= 2 && OutputTensorRank <= 4,
              "Reshape::Forward invalid input tensor rank "
                      << InputTensorRank << " or output tensor rank "
                      << OutputTensorRank);

    if constexpr (InputTensorRank == 2) {
        this->Forward(prev.GetOutput2D());
    }
    else if constexpr (InputTensorRank == 3) {
        this->Forward(prev.GetOutput3D());
    }
    else if (InputTensorRank == 4) {
        this->Forward(prev.GetOutput4D());
    }
    else {
        throw std::logic_error("Reshape::Forward UNSUPPORTED INPUT TENSOR RANK");
    }
}


template<int InputTensorRank, int OutputTensorRank>
void Reshape<InputTensorRank, OutputTensorRank>::Backward(
        const Tensor <OutputTensorRank> &gradients)
{
    fl_assert(this->Z.dimensions() == gradients.dimensions(),
              this->name << "::Backward expected gradient dimension "
                         << this->Z.dimensions() << ", instead got "
                         << gradients.dimensions());

    this->dL_dZ.resize(gradients.dimensions());
    this->dL_dZ.device(this->device) = gradients;
}


template<int InputTensorRank, int OutputTensorRank>
void Reshape<InputTensorRank, OutputTensorRank>::Backward(Layer &next)
{
    if constexpr (OutputTensorRank == 2) {
        this->dL_dZ = next.GetInputGradients2D();
    }
    else if constexpr (OutputTensorRank == 3) {
        this->dL_dZ = next.GetInputGradients3D();
    }
    else if (OutputTensorRank == 4) {
        this->dL_dZ = next.GetInputGradients4D();
    }
    else {
        throw std::logic_error("Reshape::Backward UNSUPPORTED INPUT TENSOR RANK");
    }
}


template<int InputTensorRank, int OutputTensorRank>
void Reshape<InputTensorRank, OutputTensorRank>::Update(Optimizer &)
{
    // nothing to do
}


template<int InputTensorRank, int OutputTensorRank>
const Tensor<2> &Reshape<InputTensorRank, OutputTensorRank>::GetOutput2D() const
{
    if constexpr (OutputTensorRank != 2) {
        throw std::logic_error("Reshape::GetOutput2D CALLED ON A RANK " +
                               std::to_string(OutputTensorRank) + " TENSOR");
    }

    return this->Z;
}


template<int InputTensorRank, int OutputTensorRank>
const Tensor<3> &Reshape<InputTensorRank, OutputTensorRank>::GetOutput3D() const
{
    if constexpr (OutputTensorRank != 3) {
        throw std::logic_error("Reshape::GetOutput3D CALLED ON A RANK " +
                               std::to_string(OutputTensorRank) + " TENSOR");
    }

    return this->Z;
}


template<int InputTensorRank, int OutputTensorRank>
const Tensor<4> &Reshape<InputTensorRank, OutputTensorRank>::GetOutput4D() const
{
    if constexpr (OutputTensorRank != 4) {
        throw std::logic_error("Reshape::GetOutput4D CALLED ON A RANK " +
                               std::to_string(OutputTensorRank) + " TENSOR");
    }

    return this->Z;
}


template<int InputTensorRank, int OutputTensorRank>
const Tensor<2> &Reshape<InputTensorRank, OutputTensorRank>::GetInputGradients2D()
{
    if constexpr (InputTensorRank != 2) {
        throw std::logic_error(
                "Reshape::GetInputGradients2D CALLED ON A RANK " +
                std::to_string(InputTensorRank) + " TENSOR");
    }

    return this->dL_dZ.reshape(this->input_dims);
}


template<int InputTensorRank, int OutputTensorRank>
const Tensor<3> &Reshape<InputTensorRank, OutputTensorRank>::GetInputGradients3D()
{
    if constexpr (InputTensorRank != 3) {
        throw std::logic_error(
                "Reshape::GetInputGradients3D CALLED ON A RANK " +
                std::to_string(InputTensorRank) + " TENSOR");
    }

    return this->dL_dZ.reshape(input_dims);
}


template<int InputTensorRank, int OutputTensorRank>
const Tensor<4> &Reshape<InputTensorRank, OutputTensorRank>::GetInputGradients4D()
{
    if constexpr (InputTensorRank != 4) {
        throw std::logic_error(
                "Reshape::GetInputGradients4D CALLED ON A RANK " +
                std::to_string(InputTensorRank) + " TENSOR");
    }

    return this->dL_dZ.reshape(input_dims);
}


template<int InputTensorRank, int OutputTensorRank>
int Reshape<InputTensorRank, OutputTensorRank>::GetInputRank() const
{
    return InputTensorRank;
}


template<int InputTensorRank, int OutputTensorRank>
int Reshape<InputTensorRank, OutputTensorRank>::GetOutputRank() const
{
    return OutputTensorRank;
}

} // namespace fl