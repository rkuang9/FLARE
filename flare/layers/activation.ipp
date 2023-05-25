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
    this->input_rank = TensorRank;
    this->output_rank = TensorRank;
}


template<typename activation, int TensorRank>
void Activation<activation, TensorRank>::Forward(const Tensor <TensorRank> &inputs)
{
    this->X.resize(inputs.dimensions());
    this->X.device(this->device) = inputs;
    this->Z.resize(inputs.dimensions());
    this->Z.device(this->device) = activation::Activate(inputs);
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
        const Tensor <TensorRank> &gradients)
{
    fl_assert(this->Z.dimensions() == gradients.dimensions(),
              this->name << "::Backward expected gradient dimension "
                         << this->Z.dimensions() << ", instead got "
                         << gradients.dimensions());
    this->dL_dZ.resize(gradients.dimensions());
    this->dL_dZ.device(this->device) = gradients;
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
    else if constexpr(std::is_same_v<activation, Softmax>) {
        Tensor<3> softmax_grad(this->Z.dimension(0), this->Z.dimension(1),
                               this->Z.dimension(1));
        softmax_grad = Softmax::Gradients(this->Z);

        this->dL_dX.resize(this->X.dimensions());

        ContractDim matmul {Axes(0, 1)};

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) default(none) shared(softmax_grad, matmul)
#endif
        for (int a = 0; a < this->dL_dZ.dimension(0); a++) {
            this->dL_dX.chip(a, 0) = this->dL_dZ.chip(a, 0)
                    .contract(softmax_grad.chip(a, 0), matmul);
        }
    }
    else {
        // forward pass: z = g(x)
        // backward pass (input): dL/dX = dL/dZ * dZ/dX = dL/dZ * g'(x)
        this->dL_dX.resize(this->X.dimensions());
        this->dL_dX.template device(this->device) =
                this->dL_dZ * activation::Gradients(this->X);
    }

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
    else if constexpr(std::is_same_v<activation, Softmax>) {
        Tensor<4> softmax_grad(this->Z.dimension(0), this->Z.dimension(1),
                               this->Z.dimension(2), this->Z.dimension(2));
        softmax_grad = Softmax::Gradients(this->Z);

        this->dL_dX.resize(this->X.dimensions());

        ContractDim matmul {Axes(0, 1)};

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) default(none) shared(softmax_grad, matmul)
#endif
        for (int a = 0; a < this->dL_dZ.dimension(0); a++) {
            for (int b = 0; b < this->dL_dZ.dimension(1); b++) {
                this->dL_dX.chip(a, 0).chip(b, 0) =
                        this->dL_dZ.chip(a, 0).chip(b, 0)
                                .contract(softmax_grad.chip(a, 0).chip(b, 0),
                                          matmul);

            }
        }
    }
    else {
        this->dL_dX.resize(this->X.dimensions());
        this->dL_dX.template device(this->device) =
                this->dL_dZ * activation::Gradients(this->X);
    }

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
    else if constexpr(std::is_same_v<activation, Softmax>) {
        Tensor<5> softmax_grad(this->Z.dimension(0), this->Z.dimension(1),
                               this->Z.dimension(2), this->Z.dimension(3),
                               this->Z.dimension(3));
        softmax_grad = Softmax::Gradients(this->Z);

        this->dL_dX.resize(this->X.dimensions());

        ContractDim matmul {Axes(0, 1)};

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) default(none) shared(softmax_grad, matmul)
#endif
        for (int a = 0; a < this->dL_dZ.dimension(0); a++) {
            for (int b = 0; b < this->dL_dZ.dimension(1); b++) {
                for (int c = 0; c < this->dL_dZ.dimension(2); c++) {
                    this->dL_dX.chip(a, 0).chip(b, 0).chip(c, 0) =
                            this->dL_dZ.chip(a, 0)
                                    .chip(b, 0)
                                    .chip(c, 0)
                                    .contract(softmax_grad
                                                      .chip(a, 0)
                                                      .chip(b, 0)
                                                      .chip(c, 0), matmul);
                }
            }
        }
    }
    else {
        this->dL_dX.resize(this->X.dimensions());
        this->dL_dX.template device(this->device) =
                this->dL_dZ * activation::Gradients(this->X);
    }

    return this->dL_dX;
}

} // namespace fl