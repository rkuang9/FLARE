//
// Created by R on 4/11/23.
//

#ifndef FLARE_CATEGORICAL_ACCURACY_HPP
#define FLARE_CATEGORICAL_ACCURACY_HPP

#include "metric.hpp"

namespace fl
{

template<int TensorRank>
class CategoricalAccuracy : public Metric<TensorRank>
{
public:
    explicit CategoricalAccuracy(Scalar threshold = 0.5) : threshold(threshold)
    {
        this->name = "categorical_accuracy";
    }


    void operator()(const Tensor<TensorRank> &label,
                    const Tensor<TensorRank> &pred) override
    {
        FL_REQUIRES(
                pred.dimensions() == label.dimensions(),
                this->name << " pred dimensions " << pred.dimensions()
                           << " != label dimensions " << label.dimensions());

        this->total = label.size() / label.dimension(TensorRank - 1);

        Tensor<TensorRank> pred_rounded(pred.dimensions());
        pred_rounded.device(this->device) = (pred <= this->threshold).select(
                pred.constant(0), pred.constant(1));

        Tensor<TensorRank> label_rounded(label.dimensions());
        label_rounded.device(this->device) = label.round();


        this->correct += Tensor<0, Scalar>(
                (pred_rounded == label_rounded) // compare tensors
                        .all(Dims<1>(TensorRank - 1)) // on the last dim
                        .template cast<Scalar>() // cast from bool back to Scalar
                        .sum())(0); // resultant tensor indicates pred matching label
    }


    double GetMetric() const override
    {
        return static_cast<double>(this->correct / this->total);
    }


    void Reset() override
    {
        this->total = 0;
        this->correct = 0;
    }


private:
    Scalar threshold;

    Scalar correct = 0;
    Scalar total = 0;
};

} // namespace fl

#endif //FLARE_CATEGORICAL_ACCURACY_HPP
