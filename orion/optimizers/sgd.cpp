//
// Created by macross on 8/8/22.
//

#include "sgd.hpp"

namespace orion
{

SGD::SGD(Scalar learning_rate, Scalar momentum) : Optimizer(learning_rate),
                                                  momentum(momentum)
{
}


void SGD::Minimize(Tensor2D &W, const Tensor2D &dL_dW)
{
    Tensor2D &velocity = this->v_dw[W.data()];

    if (velocity.size() == 0) {
        // on first run, initialize zero matrix with same shape as weights
        velocity.resize(W.dimensions());
        velocity.setZero();
    }

    velocity = this->momentum * velocity + dL_dW; // TensorFlow's version
    //velocity = this->momentum * velocity + (1 - this->momentum) * dL_dW; // Andrew Ng's version

    W -= this->learning_rate * velocity;
}


void SGD::Step()
{

}


void SGD::Minimize(Tensor1D &b, const Tensor1D &dL_db)
{
    Tensor1D &velocity = this->v_db[b.data()];

    if (velocity.size() == 0) {
        // on first run, initialize zero matrix with same shape as bias
        velocity.resize(b.dimensions());
        velocity.setZero();
    }

    velocity = this->momentum * velocity + dL_db; // TensorFlow's version
    //velocity = this->momentum * velocity + (1 - this->momentum) * dL_db; // Andrew Ng's version

    b -= this->learning_rate * velocity;
}


Scalar SGD::GetMomentum() const
{
    return this->momentum;
}

} // namespace orion