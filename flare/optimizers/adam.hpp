//
// Created by macross on 8/8/22.
//

#ifndef FLARE_ADAM_HPP
#define FLARE_ADAM_HPP

#include "optimizer.hpp"

namespace fl
{

// https://arxiv.org/abs/1412.6980
// applies optimization variant from the end of the paper's section 2
class Adam : public Optimizer
{
public:
    explicit Adam(Scalar learning_rate = 0.001, Scalar beta1 = 0.9,
                  Scalar beta2 = 0.999)
            : Optimizer(learning_rate),
              beta1(beta1), beta2(beta2),
              beta1_t(beta1), beta2_t(beta2),
              lr_t(this->learning_rate * std::sqrt(1 - this->beta2_t) /
                   (1 - this->beta1_t))
    {
        // initialized with time t=1
    }


    void Step() override
    {
        this->beta1_t *= this->beta1;
        this->beta2_t *= this->beta2;

        this->lr_t = this->learning_rate * std::sqrt(1 - this->beta2_t) /
                     (1 - this->beta1_t);
    }


    void Minimize(Tensor<1> &weights, const Tensor<1> &gradients) override
    {
        Tensor<1> &momentum = this->momentum_db[weights.data()];
        Tensor<1> &rmsprop = this->rmsprop_db[weights.data()];

        this->Update(weights, gradients, momentum, rmsprop);
    }


    void Minimize(Tensor<2> &weights, const Tensor<2> &gradients) override
    {
        Tensor<2> &momentum = this->momentum_dw[weights.data()];
        Tensor<2> &rmsprop = this->rmsprop_dw[weights.data()];

        this->Update(weights, gradients, momentum, rmsprop);
    }


    void Minimize(Tensor<3> &weights, const Tensor<3> &gradients) override
    {
        Tensor<3> &momentum = this->momentum_dw3[weights.data()];
        Tensor<3> &rmsprop = this->rmsprop_dw3[weights.data()];

        this->Update(weights, gradients, momentum, rmsprop);
    }


    void Minimize(Tensor<4> &kernels, const Tensor<4> &gradients) override
    {
        Tensor<4> &momentum = this->momentum_dk[kernels.data()];
        Tensor<4> &rmsprop = this->rmsprop_dk[kernels.data()];

        this->Update(kernels, gradients, momentum, rmsprop);
    }


private:
    template<int TensorRank>
    void Update(
            Tensor<TensorRank> &weights, const Tensor<TensorRank> &gradients,
            Tensor<TensorRank> &momentum, Tensor<TensorRank> &rmsprop)
    {
        // on first run, initialize zero matrix with same shape as weights
        if (momentum.size() == 0) {
            momentum.resize(weights.dimensions());
            momentum.setZero();
        }

        if (rmsprop.size() == 0) {
            rmsprop.resize(weights.dimensions());
            rmsprop.setZero();
        }

        momentum.device(this->device) =
                this->beta1 * momentum +
                (1 - this->beta1) * gradients; // * (1 - momentum)

        rmsprop.device(this->device) =
                this->beta2 * rmsprop +
                (1 - this->beta2) * gradients.square();

        weights.device(this->device) -=
                this->lr_t * momentum / (rmsprop.sqrt() + this->epsilon);
    }


    Scalar beta1; // momentum
    Scalar beta2; // RMSprop

    Scalar beta1_t; // beta1 at time step t
    Scalar beta2_t; // beta2 at time step t
    Scalar lr_t; // learning rate at time step t

    Scalar epsilon = 1e-7; // numeric stability constant

    // holds moving averages per layer, stored using Tensor.data() pointer as key
    // is unordered_map faster?
    std::map<const Scalar *, Tensor<1>> momentum_db;
    std::map<const Scalar *, Tensor<1>> rmsprop_db;

    std::map<const Scalar *, Tensor<2>> momentum_dw;
    std::map<const Scalar *, Tensor<2>> rmsprop_dw;

    std::map<const Scalar *, Tensor<3>> momentum_dw3;
    std::map<const Scalar *, Tensor<3>> rmsprop_dw3;

    std::map<const Scalar *, Tensor<4>> momentum_dk;
    std::map<const Scalar *, Tensor<4>> rmsprop_dk;
};

} // namespace fl

#endif //FLARE_ADAM_HPP
