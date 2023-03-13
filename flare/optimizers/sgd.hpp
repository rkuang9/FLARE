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
    explicit SGD(Scalar learning_rate = 0.01, Scalar momentum = 0);

    ~SGD() = default;

    void Step() override;

    void Minimize(Tensor<1> &weights, const Tensor<1> &gradients) override;

    void Minimize(Tensor<2> &weights, const Tensor<2> &gradients) override;

    void Minimize(Tensor<3> &weights, const Tensor<3> &gradients) override;

    void Minimize(Tensor<4> &kernels, const Tensor<4> &gradients) override;

private:
    template<int TensorRank>
    void Update(Tensor<TensorRank> &weights,
                const Tensor<TensorRank> &gradients,
                Tensor<TensorRank> &velocity);

    Scalar momentum = 0; // default 0 means no momentum

    // holds moving averages per layer, stored using Tensor.data() pointer as key
    std::map<const Scalar *, Tensor<1>> v_db;
    std::map<const Scalar *, Tensor<2>> v_dw;
    std::map<const Scalar *, Tensor<3>> v_dy;
    std::map<const Scalar *, Tensor<4>> v_dk;
};

} // namespace fl

#endif //FLARE_SGD_HPP
