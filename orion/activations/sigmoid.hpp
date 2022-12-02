//
// Created by macross on 8/7/22.
//

#ifndef ORION_SIGMOID_HPP
#define ORION_SIGMOID_HPP

#include "orion/orion_types.hpp"

namespace orion
{

class Sigmoid
{
public:
    // tensorflow automatically swaps sigmoid with softmax if categorical cross
    // entropy is used (tf.nn.sigmoid prevents this), we will not do that here
    template<int TensorRank>
    static Tensor<TensorRank> Activate(const Tensor<TensorRank> &features)
    {
        return features.sigmoid();
    }


    template<int TensorRank>
    static Tensor<TensorRank> Gradients(const Tensor<TensorRank> &features)
    {
        auto one = static_cast<Scalar>(1.0);

        Tensor<TensorRank> sigmoid = features.sigmoid();
        return sigmoid * (one - sigmoid);
    }
};

} // namespace orion

#endif //ORION_SIGMOID_HPP
