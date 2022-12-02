//
// Created by Macross on 11/25/22.
//

#include "categorical_cross_entropy.hpp"

namespace orion
{

CategoricalCrossEntropy::CategoricalCrossEntropy(Scalar epsilon, int history_size)
        : LossFunction(epsilon)
{

}


void CategoricalCrossEntropy::CalculateLoss(const Tensor<2> &predict,
                                            const Tensor<2> &label)
{
    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history2D.push_back(this->Gradient(predict, label));
}


void CategoricalCrossEntropy::CalculateLoss(const Tensor<4> &predict,
                                            const Tensor<4> &label)
{
    this->loss_history.push_back(this->Loss(predict, label));
    this->gradient_history4D.push_back(this->Gradient(predict, label));
}


template<int TensorRank>
Scalar CategoricalCrossEntropy::Loss(const Tensor<TensorRank> &predict,
                                     const Tensor<TensorRank> &label)
{
    // see https://github.com/keras-team/keras/blob/985521ee7050df39f9c06f53b54e17927bd1e6ea/keras/backend/numpy_backend.py#L333
    // cross entropy goes as follows using example predict with dims [2, 3, 4, 5]:
    // 1. find the sum of predict along the last dim, returns [2, 3, 4]
    // 2. reshape back to [2, 3, 4, 1]
    // 3. broadcast it back to [2, 3, 4, 5] and divide predict with it
    // 4. apply the formula sum(-label * ln(predict)) to get a scalar value
    // 5. divide the result by number of values summed over, 24 which is 2*3*4

    // get the summed predict tensor back to the same rank as predict
    Dims<TensorRank> predict_summed = predict.dimensions();
    predict_summed.back() = 1;

    // broadcast dims will be all ones except the last dim (which was summed over)
    Dims<TensorRank> bcast;
    bcast.fill(1);
    bcast.back() = predict.dimension(TensorRank - 1);

    auto predict_norm = predict / predict.sum(Dims<1>(TensorRank - 1)).reshape(
            predict_summed).broadcast(bcast);

    return Tensor<0>(
            (-label * predict_norm.clip(this->clip_min, this->clip_max).log())
                    .sum()).coeff() /
           (predict.dimensions().TotalSize() / predict.dimension(TensorRank - 1));
}


template<int TensorRank>
Tensor<TensorRank> CategoricalCrossEntropy::Gradient(
        const Tensor<TensorRank> &predict,
        const Tensor<TensorRank> &label)
{
    Dims<TensorRank> predict_sum_dims = predict.dimensions();
    predict_sum_dims.back() = 1;
    Dims<TensorRank> label_sum_dims = predict_sum_dims;

    Dims<TensorRank> bcast;
    bcast.fill(1);
    bcast.back() = predict.dimension(TensorRank - 1);

    auto label_predict_sum =
            label
                    .sum(Dims<1>(TensorRank - 1))
                    .eval()
                    .reshape(label_sum_dims)
                    .broadcast(bcast) /
            predict
                    .sum(Dims<1>(TensorRank - 1))
                    .eval()
                    .reshape(predict_sum_dims)
                    .broadcast(bcast);

    return (-label / predict + label_predict_sum);
}

} // namespace orion