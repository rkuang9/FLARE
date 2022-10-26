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


void SGD::Minimize(Tensor<2> &W, const Tensor<2> &dL_dW)
{
    Tensor<2> &velocity = this->v_dw[W.data()];

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


void SGD::Minimize(Tensor<1> &b, const Tensor<1> &dL_db)
{
    Tensor<1> &velocity = this->v_db[b.data()];

    if (velocity.size() == 0) {
        // on first run, initialize zero matrix with same shape as bias
        velocity.resize(b.dimensions());
        velocity.setZero();
    }

    velocity = this->momentum * velocity + dL_db; // TensorFlow's version
    //velocity = this->momentum * velocity + (1 - this->momentum) * dL_db; // Andrew Ng's version

    b -= this->learning_rate * velocity;
}


void SGD::Minimize(Tensor<4> &k, const Tensor<4> &dL_dk)
{
    Tensor<4> &velocity = this->v_dk[k.data()];

    if (velocity.size() == 0) {
        // on first run, initialize zero matrix with same shape as bias
        velocity.resize(k.dimensions());
        velocity.setZero();
    }

    velocity = this->momentum * velocity + dL_dk; // TensorFlow's version
    //velocity = this->momentum * velocity + (1 - this->momentum) * dL_db; // Andrew Ng's version

    k -= this->learning_rate * velocity;
}


Scalar SGD::GetMomentum() const
{
    return this->momentum;
}

} // namespace orion