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


void RMSprop::Minimize(Tensor2D &W, const Tensor2D &dL_dW)
{
    Tensor2D &velocity = this->s_dw[W.data()];

    if (velocity.size() == 0) {
        velocity.resize(W.dimensions());
        velocity.setZero();
    }

    velocity = this->momentum * velocity + (1 - this->momentum) * dL_dW * dL_dW;

    W -= this->learning_rate * dL_dW / (velocity.sqrt() + this->epsilon);
}


void RMSprop::Minimize(Tensor1D &b, const Tensor1D &dL_db)
{
    Tensor1D &velocity = this->s_db[b.data()];

    if (velocity.size() == 0) {
        velocity.resize(b.dimensions());
        velocity.setZero();
    }

    velocity = this->momentum * velocity + (1 - this->momentum) * dL_db * dL_db;

    b -= this->learning_rate * dL_db / (velocity.sqrt() + this->epsilon);
}

} // namespace orion