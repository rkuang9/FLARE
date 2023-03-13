//
// Created by macross on 8/17/22.
//

#ifndef FLARE_RMSPROP_HPP
#define FLARE_RMSPROP_HPP

#include "optimizer.hpp"

namespace fl
{

class RMSprop : public Optimizer
{
public:
    explicit RMSprop(Scalar learning_rate = 0.001, Scalar momentum = 0.9);

    void Minimize(Tensor<1> &weights, const Tensor<1> &gradients) override;

    void Minimize(Tensor<2> &weights, const Tensor<2> &gradients) override;

    void Minimize(Tensor<3> &weights, const Tensor<3> &gradients) override;

    void Minimize(Tensor<4> &kernels, const Tensor<4> &gradients) override;

    void Step() override;

private:
    template<int TensorRank>
    void Update(Tensor<TensorRank> &weights,
                          const Tensor<TensorRank> &gradients,
                          Tensor<TensorRank> &velocity);

    Scalar momentum = 0.9;
    Scalar epsilon = 1e-7; // prevent divison by zero

    std::map<const Scalar *, Tensor<1>> s_db;
    std::map<const Scalar *, Tensor<2>> s_dw;
    std::map<const Scalar *, Tensor<3>> s_dw3;
    std::map<const Scalar *, Tensor<4>> s_dk;
};

} // namespace fl

#endif //FLARE_RMSPROP_HPP
