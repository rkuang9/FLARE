//
// Created by macross on 9/5/22.
//

#ifndef ORION_MEAN_SQUARED_ERROR_HPP
#define ORION_MEAN_SQUARED_ERROR_HPP

#include "loss_function.hpp"

namespace orion
{

template<int TensorRank>
class MeanSquaredError : public LossFunction<TensorRank>
{
public:
    MeanSquaredError() = default;


    Scalar Loss(const Tensor<TensorRank> &predict, const Tensor<TensorRank> &label)
    {
        orion_assert(predict.dimensions() == label.dimensions(),
                     "predict dimensions " << predict.dimensions() <<
                                           " don't match label dimensions " <<
                                           label.dimensions());
        return Tensor<0>((label - predict).square().mean())(0);
    }


    Tensor<TensorRank> Gradient(const Tensor<TensorRank> &predict,
                                const Tensor<TensorRank> &label)
    {
        orion_assert(predict.dimensions() == label.dimensions(),
                     "predict dimensions " << predict.dimensions() <<
                                           " don't match label dimensions "
                                           << label.dimensions());
        return 2 * (predict - label) / static_cast<Scalar>(predict.size());
    }
};

} // orion

#endif //ORION_MEAN_SQUARED_ERROR_HPP
