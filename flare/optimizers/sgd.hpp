//
// Created by macross on 8/8/22.
//

#ifndef FLARE_SGD_HPP
#define FLARE_SGD_HPP

#include "optimizer.hpp"

namespace fl
{

class SGD : public Optimizer
{
public:
    explicit SGD(Scalar learning_rate = 0.01, Scalar momentum = 0)
            : Optimizer(learning_rate),
              momentum(momentum)
    {
        // nothing to do
    }


    ~SGD() = default;


    void Minimize(Tensor<1> &weights, const Tensor<1> &gradients) override
    {
        Tensor<1> &velocity = this->v_db[weights.data()];
        this->Update(weights, gradients, velocity);
    }


    void Minimize(Tensor<2> &weights, const Tensor<2> &gradients) override
    {
        Tensor<2> &velocity = this->v_dw[weights.data()];
        this->Update(weights, gradients, velocity);
    }


    void Minimize(Tensor<3> &weights, const Tensor<3> &gradients) override
    {
        Tensor<3> &velocity = this->v_dy[weights.data()];
        this->Update(weights, gradients, velocity);
    }


    void Minimize(Tensor<4> &kernels, const Tensor<4> &gradients) override
    {
        Tensor<4> &velocity = this->v_dk[kernels.data()];
        this->Update(kernels, gradients, velocity);
    }


private:
    template<int TensorRank>
    void Update(Tensor <TensorRank> &weights,
                const Tensor <TensorRank> &gradients,
                Tensor <TensorRank> &velocity)
    {
        if (velocity.size() == 0) {
            // on first run, initialize zero matrix with same shape as bias
            velocity.resize(weights.dimensions());
            velocity.setZero();
        }

        // * (1 - this->momentum)
        velocity.device(this->device) = this->momentum * velocity + gradients;

        weights.device(this->device) -= this->learning_rate * velocity;
    }


    Scalar momentum = 0; // default 0 means no momentum

    // holds moving averages per layer, stored using Tensor.data() pointer as key
    std::map<const Scalar *, Tensor < 1>> v_db;
    std::map<const Scalar *, Tensor < 2>> v_dw;
    std::map<const Scalar *, Tensor < 3>> v_dy;
    std::map<const Scalar *, Tensor < 4>> v_dk;
};

} // namespace fl

#endif //FLARE_SGD_HPP
