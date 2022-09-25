//
// Created by macross on 8/8/22.
//

#ifndef ORION_SGD_HPP
#define ORION_SGD_HPP

#include "optimizer.hpp"

namespace orion
{

class SGD : public Optimizer
{
public:
    explicit SGD(Scalar learning_rate = 0.01, Scalar momentum = 0);

    ~SGD() = default;

    void Step() override;

    void Minimize(Tensor<2> &W, const Tensor<2> &dL_dW) override;

    void Minimize(Tensor<1> &b, const Tensor<1> &dL_db) override;

    Scalar GetMomentum() const;

private:
    Scalar momentum; // default 0 means no momentum

    // holds moving averages per layer, stored using Tensor.data() pointer as key
    std::map<const Scalar *, Tensor<2>> v_dw;
    std::map<const Scalar *, Tensor<1>> v_db;
};

} // namespace orion

#endif //ORION_SGD_HPP
