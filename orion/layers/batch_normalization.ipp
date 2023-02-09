//
// Created by Macross on 12/27/22.
//

#include "batch_normalization.hpp"

namespace orion
{

template<int TensorRank, int NormDimCount>
BatchNormalization<TensorRank, NormDimCount>::BatchNormalization(
        const Dims<NormDimCount> &norm_axes, Scalar momentum,
        Scalar epsilon, bool training, bool center, bool scale):
        norm_axes(norm_axes), momentum(momentum), epsilon(epsilon),
        training_mode(training), center(center), scale(scale)
{
    this->name = "batch_norm";

    std::sort(this->norm_axes.begin(), this->norm_axes.end());

    if (this->norm_axes.back() > (TensorRank - 1)) {
        throw std::invalid_argument(
                "SPECIFIED NORM AXES EXCEED INPUT TENSOR RANK");
    }

    // populate the dimensions that will be collapsed to calculate mean and variance
    for (int i = 0, collapse_index = 0; i < TensorRank; i++) {
        if (std::find(this->norm_axes.begin(), this->norm_axes.end(), i) ==
            this->norm_axes.end()) {
            this->collapsed_dims[collapse_index] = i;
            collapse_index++;
        }
    }
}


template<int TensorRank, int NormDimCount>
void BatchNormalization<TensorRank, NormDimCount>::Forward(
        const Tensor<TensorRank> &inputs)
{
    this->X = inputs;

    // this resizing is required for multithreading
    this->X_norm.resize(inputs.dimensions());
    this->Z.resize(inputs.dimensions());
    this->input_minus_mean.resize(inputs.dimensions());
    this->variance_plus_epsilon.resize(inputs.dimensions());

    if (!this->weights_are_set) [[unlikely]] {
        Dims<NormDimCount> weight_dims;

        for (auto i = 0; i < this->norm_axes.size(); i++) {
            weight_dims[i] = inputs.dimension(this->norm_axes[i]);
        }

        this->gamma.resize(weight_dims);
        this->gamma.setConstant(1.0);

        this->beta.resize(weight_dims);
        this->beta.setZero();

        this->moving_mean.resize(weight_dims);
        this->moving_mean.setZero();

        this->moving_variance.resize(weight_dims);
        this->moving_variance.setConstant(1.0);

        this->weights_are_set = true;
    }

    this->restore_reshape = inputs.dimensions();
    this->restore_bcast = inputs.dimensions();

    for (auto i: this->norm_axes) {
        this->restore_bcast[i] = 1;
    }

    for (auto i: this->collapsed_dims) {
        this->restore_reshape[i] = 1;
    }

    if (this->training_mode) {
        // the following implements Algorithm 1 of the batch normalization paper
        // by Sergey Ioffe and Christian Szegedy

        // mini-batch mean, cached for reuse in backpropagation
        this->input_minus_mean.template device(this->device) =
                inputs - inputs.mean(this->collapsed_dims)
                        .reshape(this->restore_reshape)
                        .eval()
                        .broadcast(this->restore_bcast).eval();

        // mini-batch variance, cached for reuse in backpropagation
        this->variance_plus_epsilon.template device(this->device) =
                this->input_minus_mean
                        .square().mean(this->collapsed_dims)
                        .reshape(this->restore_reshape)
                        .eval()
                        .broadcast(this->restore_bcast).eval() + this->epsilon;

        // normalize, x_norm = (x - x_mean) / sqrt(var + epsilon)
        this->X_norm.resize(inputs.dimensions());
        this->X_norm.template device(this->device) =
                this->input_minus_mean * this->variance_plus_epsilon.rsqrt();

        // scale and shift, z = gamma * x_norm + beta
        this->Z.template device(this->device) =
                this->X_norm * this->gamma
                        .reshape(this->restore_reshape)
                        .eval()
                        .broadcast(this->restore_bcast).eval() +
                this->beta.reshape(this->restore_reshape)
                        .eval()
                        .broadcast(this->restore_bcast).eval();

        // calculate moving mean and moving variance, these are frozen during inference
        this->moving_mean.template device(this->device) =
                this->moving_mean * this->momentum +
                inputs.mean(this->collapsed_dims).eval() *
                (1.0 - this->momentum);

        this->moving_variance.template device(this->device) =
                this->moving_variance * this->momentum +
                this->input_minus_mean
                        .square()
                        .mean(this->collapsed_dims).eval() *
                (1.0 - this->momentum);
    }
    else {
        // inference mode
        // gamma * (inputs - moving_mean) / sqrt(moving_var + epsilon) + beta
        this->Z.template device(this->device) =
                this->gamma
                        .reshape(this->restore_reshape)
                        .eval()
                        .broadcast(this->restore_bcast).eval() *
                (inputs - this->moving_mean
                        .reshape(this->restore_reshape)
                        .eval()
                        .broadcast(this->restore_bcast).eval()) /
                (this->moving_variance
                         .reshape(this->restore_reshape)
                         .eval()
                         .broadcast(this->restore_bcast).eval() +
                 this->Z.constant(this->epsilon)).sqrt() +
                this->beta.reshape(this->restore_reshape)
                        .eval()
                        .broadcast(this->restore_bcast).eval();
    }
}


template<int TensorRank, int NormDimCount>
void BatchNormalization<TensorRank, NormDimCount>::Forward(const Layer &prev)
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
                "BatchNormalization::Forward expects an input tensor of rank 2, 3, or 4");
    }
}


template<int TensorRank, int NormDimCount>
void BatchNormalization<TensorRank, NormDimCount>::Backward(Layer &next)
{
    if constexpr (TensorRank == 2) {
        this->Backward(next.GetOutput2D());
    }
    else if constexpr (TensorRank == 3) {
        this->Backward(next.GetOutput3D());
    }
    else if constexpr (TensorRank == 4) {
        this->Backward(next.GetOutput4D());
    }
    else {
        throw std::logic_error("BatchNormalization::Backward " +
                               std::to_string(TensorRank) + " IS NOT SUPPORTED");
    }
}


template<int TensorRank, int NormDimCount>
void BatchNormalization<TensorRank, NormDimCount>::Backward(
        const Tensor<TensorRank> &gradients)
{
    this->dL_dZ = gradients;
    this->dL_dy = (this->dL_dZ * this->X_norm).sum(this->collapsed_dims);
    this->dL_db = this->dL_dZ.sum(this->collapsed_dims);
}


template<int TensorRank, int NormDimCount>
void BatchNormalization<TensorRank, NormDimCount>::Update(Optimizer &opt)
{
    if (this->scale) {
        opt.Minimize(this->gamma, this->dL_dy);
    }

    if (this->center) {
        opt.Minimize(this->beta, this->dL_db);
    }
}


#ifdef _WIN32
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
#elif defined __unix__ || defined __APPLE__ || defined __linux__
#pragma GCC diagnostic ignored "-Wreturn-stack-address"
#endif


template<int TensorRank, int NormDimCount>
const Tensor<2> &BatchNormalization<TensorRank, NormDimCount>::GetOutput2D() const
{
    if constexpr (TensorRank != 2) {
        throw std::logic_error("BatchNormalization::GetOutput2D CALLED ON A RANK " +
                               std::to_string(TensorRank) + " TENSOR");
    }

    return this->Z;
}


#ifdef _WIN32
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
#elif defined __unix__ || defined __APPLE__ || defined __linux__
#pragma GCC diagnostic ignored "-Wreturn-stack-address"
#endif


template<int TensorRank, int NormDimCount>
const Tensor<3> &BatchNormalization<TensorRank, NormDimCount>::GetOutput3D() const
{
    if constexpr (TensorRank != 3) {
        throw std::logic_error("BatchNormalization::GetOutput3D CALLED ON A RANK " +
                               std::to_string(TensorRank) + " TENSOR");
    }

    return this->Z;
}


#ifdef _WIN32
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
#elif defined __unix__ || defined __APPLE__ || defined __linux__
#pragma GCC diagnostic ignored "-Wreturn-stack-address"
#endif


template<int TensorRank, int NormDimCount>
const Tensor<4> &BatchNormalization<TensorRank, NormDimCount>::GetOutput4D() const
{
    if constexpr (TensorRank != 4) {
        throw std::logic_error("BatchNormalization::GetOutput4D CALLED ON A RANK " +
                               std::to_string(TensorRank) + " TENSOR");
    }

    return this->Z;
}


#ifdef _WIN32
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
#elif defined __unix__ || defined __APPLE__ || defined __linux__
#pragma GCC diagnostic ignored "-Wreturn-stack-address"
#endif


template<int TensorRank, int NormDimCount>
const Tensor<2> &
BatchNormalization<TensorRank, NormDimCount>::GetInputGradients2D()
{
    if constexpr (TensorRank != 2) {
        throw std::logic_error(
                "BatchNormalization::GetInputGradients2D CALLED ON A RANK " +
                std::to_string(TensorRank) + " TENSOR");
    }

    this->dL_dX.resize(this->X.dimensions());
    this->CalculateInputGradients(this->dL_dZ);
    return this->dL_dX;
}


#ifdef _WIN32
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
#elif defined __unix__ || defined __APPLE__ || defined __linux__
#pragma GCC diagnostic ignored "-Wreturn-stack-address"
#endif


template<int TensorRank, int NormDimCount>
const Tensor<3> &
BatchNormalization<TensorRank, NormDimCount>::GetInputGradients3D()
{
    if constexpr (TensorRank != 3) {
        throw std::logic_error(
                "BatchNormalization::GetInputGradients3D CALLED ON A RANK " +
                std::to_string(TensorRank) + " TENSOR");
    }

    this->dL_dX.resize(this->X.dimensions());
    this->CalculateInputGradients(this->dL_dZ);
    return this->dL_dX;
}


#ifdef _WIN32
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
#elif defined __unix__ || defined __APPLE__ || defined __linux__
#pragma GCC diagnostic ignored "-Wreturn-stack-address"
#endif


template<int TensorRank, int NormDimCount>
const Tensor<4> &
BatchNormalization<TensorRank, NormDimCount>::GetInputGradients4D()
{
    if constexpr (TensorRank != 4) {
        throw std::logic_error(
                "BatchNormalization::GetInputGradients4D CALLED ON A RANK " +
                std::to_string(TensorRank) + " TENSOR");
    }

    this->dL_dX.resize(this->X.dimensions());
    this->CalculateInputGradients(this->dL_dZ);
    return this->dL_dX;
}


template<int TensorRank, int NormDimCount>
int BatchNormalization<TensorRank, NormDimCount>::GetInputRank() const
{
    return TensorRank;
}


template<int TensorRank, int NormDimCount>
int BatchNormalization<TensorRank, NormDimCount>::GetOutputRank() const
{
    return TensorRank;
}


template<int TensorRank, int NormDimCount>
void BatchNormalization<TensorRank, NormDimCount>::Training(bool is_training)
{
    this->training_mode = is_training;
}


template<int TensorRank, int NormDimCount>
void BatchNormalization<TensorRank, NormDimCount>::CalculateInputGradients(
        const Tensor<TensorRank> &gradients)
{
    Scalar collapsed_dim_size = 1;

    for (auto i: this->collapsed_dims) {
        collapsed_dim_size *= gradients.dimension(i);
    }

    auto norm_grad = gradients * this->gamma
            .reshape(this->restore_reshape)
            .eval()
            .broadcast(this->restore_bcast).eval();
    auto var_grad = (norm_grad * this->input_minus_mean * -0.5 *
                     this->variance_plus_epsilon.pow(-1.5))
            .sum(this->collapsed_dims)
            .reshape(this->restore_reshape)
            .eval()
            .broadcast(this->restore_bcast).eval();
    auto mean_grad =
            (norm_grad * -this->variance_plus_epsilon.rsqrt())
                    .sum(this->collapsed_dims)
                    .reshape(this->restore_reshape)
                    .eval()
                    .broadcast(this->restore_bcast).eval() -
            2.0 * var_grad *
            (this->input_minus_mean.mean(this->collapsed_dims)
                    .reshape(this->restore_reshape)
                    .eval()
                    .broadcast(this->restore_bcast).eval());

    this->dL_dX.template device(this->device) =
            norm_grad * this->variance_plus_epsilon.rsqrt() +
            var_grad * static_cast<Scalar>(2.0 / collapsed_dim_size) *
            this->input_minus_mean +
            mean_grad / collapsed_dim_size;

}


template<int TensorRank, int NormDimCount>
void BatchNormalization<TensorRank, NormDimCount>::Save(const std::string &path)
{
    throw std::logic_error(this->name + "::Save not implemented yet");
}


template<int TensorRank, int NormDimCount>
void BatchNormalization<TensorRank, NormDimCount>::Load(const std::string &path)
{
    throw std::logic_error(this->name + "::Load not implemented yet");
}

} // namespace orion