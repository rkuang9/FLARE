//
// Created by macross on 8/8/22.
//

#include "adam.hpp"

namespace orion
{

Adam::Adam(Scalar learning_rate, Scalar beta1, Scalar beta2)
        : Optimizer(learning_rate),
          beta1(beta1), beta2(beta2),
          beta1_correction(beta1), beta2_correction(beta2)
{

}


// https://arxiv.org/pdf/1412.6980.pdf, applies optimization variant in section 2
void Adam::Minimize(Tensor2D &W, const Tensor2D &g)
{
    Tensor2D &m = this->momentum[W.data()]; // momentum
    Tensor2D &v = this->rmsprop[W.data()]; // RMSprop

    // on first run, initialize zero matrix with same shape as weights
    if (m.size() == 0) {
        m.resize(W.dimensions());
        m.setZero();
    }

    if (v.size() == 0) {
        v.resize(W.dimensions());
        v.setZero();
    }

    m = this->beta1 * m + g; // momentum
    v = this->beta2 * v + (1 - this->beta2) * g * g; // RMSprop

    // bias correction
    //v_dw = v_dw / (1 - this->beta1_correction);
    //s_dw = s_dw / (1 - this->beta2_correction);
    this->learning_rate = this->learning_rate *
                          std::sqrt(1 - this->beta2_correction) /
                          (1 - this->beta1_correction);

    W -= this->learning_rate * m / (v.sqrt() + this->epsilon);

    this->beta1_correction *= this->beta1;
    this->beta2_correction *= this->beta2;
}


void Adam::Minimize(Tensor1D &b, const Tensor1D &dL_db)
{
    throw std::logic_error("BIAS OPTIMIZATION NOT IMPLEMENTED YET");
}

} // namespace orion