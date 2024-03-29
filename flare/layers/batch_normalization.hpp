//
// Created by Macross on 12/27/22.
//

#ifndef FLARE_BATCH_NORMALIZATION_H
#define FLARE_BATCH_NORMALIZATION_H

#include "layer.hpp"

namespace fl
{

template<int TensorRank, int NormDimCount>
class BatchNormalization : public Layer
{
public:
    explicit BatchNormalization(
            const Dims<NormDimCount> &norm_axes, Scalar momentum = 0.99,
            Scalar epsilon = 0.001, bool training = false, bool center = true,
            bool scale = true);

    void Forward(const Tensor<TensorRank> &inputs) override;

    void Forward(const Layer &prev) override;

    void Backward(const Tensor<TensorRank> &gradients) override;

    void Backward(Layer &next) override;

    void Update(Optimizer &opt) override;

    const Tensor<2> &GetOutput2D() const override;

    const Tensor<3> &GetOutput3D() const override;

    const Tensor<4> &GetOutput4D() const override;

    const Tensor<2> &GetInputGradients2D() override;

    const Tensor<3> &GetInputGradients3D() override;

    const Tensor<4> &GetInputGradients4D() override;

    int GetInputRank() const override;

    int GetOutputRank() const override;

    void Training(bool is_training) override;

    void Save(const std::string &path) override;

    void Load(const std::string &path) override;

    Tensor<NormDimCount> moving_mean;
    Tensor<NormDimCount> moving_variance;
    Tensor<NormDimCount> beta;
    Tensor<NormDimCount> gamma;

private:
    void CalculateInputGradients(const Tensor<TensorRank> &gradients);

    Tensor<TensorRank> X; // inputs
    Tensor<TensorRank> X_norm;
    Tensor<TensorRank> Z; // outputs

    Tensor<TensorRank> dL_dZ;   // output gradients from next layer or a loss function
    Tensor<NormDimCount> dL_db; // beta gradients w.r.t. loss
    Tensor<NormDimCount> dL_dy; // gamma gradients w.r.t. loss
    Tensor<TensorRank> dL_dX;   // layer input gradients
    bool weights_are_set = false;

    // hyperparameters
    Dims<NormDimCount> norm_axes;
    Scalar momentum = 0.99;
    Scalar epsilon = 0.001;
    bool center = true;
    bool scale = true;

    // cached values calculated in forward, reused in backward to save time
    Tensor<TensorRank> input_minus_mean;
    Tensor<TensorRank> variance_plus_epsilon;

    // during mean calculation, collapsed_dims (dims not specified in norm_axes)
    // go away, restore_reshape and restore_bcast will restore the mean tensor back
    // into the input tensor dims
    Dims<TensorRank> restore_reshape;
    Dims<TensorRank> restore_bcast;
    Dims<TensorRank - NormDimCount> collapsed_dims; // dimensions not in norm_axes

    // multithreading
    Eigen::ThreadPool pool = Eigen::ThreadPool((int) std::thread::hardware_concurrency());
    Eigen::ThreadPoolDevice device = Eigen::ThreadPoolDevice(&pool, 2);

    // false for inference, true for training
    bool training_mode = false;
};

} // namespace fl

#include "batch_normalization.ipp"

#endif //FLARE_BATCH_NORMALIZATION_H
