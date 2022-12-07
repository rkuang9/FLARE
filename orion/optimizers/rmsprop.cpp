//
// Created by macross on 8/17/22.
//

#include "rmsprop.hpp"

namespace orion
{

RMSprop::RMSprop(Scalar learning_rate, Scalar momentum) :
        Optimizer(learning_rate), momentum(momentum)
{

}


void RMSprop::Minimize(Tensor<2> &W, const Tensor<2> &dL_dW)
{
    Tensor<2> &velocity = this->s_dw[W.data()];

    if (velocity.size() == 0) {
        velocity.resize(W.dimensions());
        velocity.setZero();
    }

    velocity = this->momentum * velocity + (1 - this->momentum) * dL_dW * dL_dW;

    W -= this->learning_rate * dL_dW / (velocity.sqrt() + this->epsilon);
}


void RMSprop::Minimize(Tensor<1> &b, const Tensor<1> &dL_db)
{
    Tensor<1> &velocity = this->s_db[b.data()];

    if (velocity.size() == 0) {
        velocity.resize(b.dimensions());
        velocity.setZero();
    }

    velocity = this->momentum * velocity + (1 - this->momentum) * dL_db * dL_db;

    b -= this->learning_rate * dL_db / (velocity.sqrt() + this->epsilon);
}

void RMSprop::Step()
{

}

} // namespace orion