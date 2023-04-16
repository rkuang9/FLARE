//
// Created by R on 4/11/23.
//

#ifndef FLARE_CATEGORICAL_ACCURACY_HPP
#define FLARE_CATEGORICAL_ACCURACY_HPP

#include "metric.hpp"

namespace fl
{

// Computes the frequency in which one-hot vector predictions match the labels
template<int TensorRank>
class CategoricalAccuracy : public Metric<TensorRank>
{
public:
    explicit CategoricalAccuracy(Scalar threshold = 0.5) : threshold(threshold)
    {
        this->name = "categorical_accuracy";
    }


    void operator()(const Tensor <TensorRank> &pred,
                    const Tensor <TensorRank> &label) override
    {
        FL_REQUIRES(
                pred.dimensions() == label.dimensions(),
                this->name << " pred dimensions " << pred.dimensions()
                           << " != label dimensions " << label.dimensions());

        this->total += label.size() / label.dimension(TensorRank - 1);

        // use argmax to find index (first, if multiple) of the largest value on
        // the last dimension which is assumed to contain the softmax prediction,
        Tensor<0> equals;
        equals.device(this->device) =
                (pred.argmax(TensorRank - 1) == label.argmax(TensorRank - 1))
                        .template cast<Scalar>() // equality tensor rank is 1 less
                        .sum(); // cast from bool back to Scalar, sum for # correct
        this->correct += equals.coeff();
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
