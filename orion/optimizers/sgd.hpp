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

    void Minimize(Tensor2D &W, const Tensor2D &dL_dW) override;

    void Minimize(Tensor1D &b, const Tensor1D &dL_db) override;

    Scalar GetMomentum() const;

private:
    Scalar momentum; // default 0 means no momentum

    // holds moving averages per layer, stored using Tensor.data() pointer as key
    std::map<const Scalar *, Tensor2D> v_dw;
    std::map<const Scalar *, Tensor1D> v_db;
};

} // namespace orion

#endif //ORION_SGD_HPP
