//
// Created by macross on 8/8/22.
//

#ifndef ORION_ADAM_HPP
#define ORION_ADAM_HPP

#include "optimizer.hpp"

namespace orion
{

class Adam : public Optimizer
{
public:
    explicit Adam(Scalar learning_rate = 0.001, Scalar beta1 = 0.9,
                  Scalar beta2 = 0.999);

    void Minimize(Tensor2D &W, const Tensor2D &dL_dw) override;

    void Minimize(Tensor1D &b, const Tensor1D &dL_db) override;

private:
    Scalar beta1; // momentum
    Scalar beta2; // RMSprop

    Scalar beta1_correction; // bias correction
    Scalar beta2_correction; // bias correction

    Scalar epsilon = 1e-7; // numeric stability constant

    // holds moving averages per layer, stored using Tensor.data() pointer as key
    // is unordered_map faster?
    std::map<const Scalar *, Tensor2D> momentum;
    std::map<const Scalar *, Tensor1D> v_db;

    std::map<const Scalar *, Tensor2D> rmsprop;
    std::map<const Scalar *, Tensor1D> s_db;
};

} // namespace orion

#endif //ORION_ADAM_HPP
