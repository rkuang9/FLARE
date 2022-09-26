//
// Created by macross on 8/8/22.
//

#include "adam.hpp"

namespace orion
{

Adam::Adam(Scalar learning_rate, Scalar beta1, Scalar beta2)
        : Optimizer(learning_rate),
          beta1(beta1), beta2(beta2),
          beta1_t(beta1), beta2_t(beta2),
          lr_t(this->learning_rate * std::sqrt(1 - this->beta2_t) /
               (1 - this->beta1_t))
{
}


void Adam::Step()
{
    this->beta1_t *= this->beta1;
    this->beta2_t *= this->beta2;

    this->lr_t = this->learning_rate * std::sqrt(1 - this->beta2_t) /
                 (1 - this->beta1_t);
}


// applies optimization variant in section 2 of the paper
void Adam::Minimize(Tensor<2> &w, const Tensor<2> &g)
{
    Tensor<2> &m = this->momentum[w.data()]; // momentum
    Tensor<2> &v = this->rmsprop[w.data()]; // RMSprop

    // on first run, initialize zero matrix with same shape as weights
    if (m.size() == 0) {
        m.resize(w.dimensions());
        m.setZero();
    }

    if (v.size() == 0) {
        v.resize(w.dimensions());
        v.setZero();
    }

    m = this->beta1 * m + (1 - this->beta1) * g; // momentum
    v = this->beta2 * v + (1 - this->beta2) * g.square(); // RMSprop

    w -= this->lr_t * m / (v.sqrt() + this->epsilon);
}


void Adam::Minimize(Tensor<1> &b, const Tensor<1> &dL_db)
{
    throw std::logic_error("BIAS OPTIMIZATION NOT IMPLEMENTED YET");
}

} // namespace orion