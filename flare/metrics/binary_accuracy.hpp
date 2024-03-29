//
// Created by R on 4/9/23.
//

#ifndef FLARE_BINARY_ACCURACY_HPP
#define FLARE_BINARY_ACCURACY_HPP

#include "metric.hpp"

namespace fl
{

// Computes the frequency in which binary predictions match the labels
template<int TensorRank>
class BinaryAccuracy : public Metric<TensorRank>
{
public:
    explicit BinaryAccuracy(Scalar threshold = 0.5) : threshold(threshold)
    {
        this->name = "binary_accuracy";
    }


    void operator()(const Tensor <TensorRank> &pred,
                    const Tensor <TensorRank> &label) override
    {
        FL_REQUIRES(
                pred.dimensions() == label.dimensions(),
                this->name << " pred dimensions " << pred.dimensions()
                           << " != label dimensions " << label.dimensions());

        this->total += label.size();

        Tensor<TensorRank> pred_rounded(pred.dimensions());
        pred_rounded.device(this->device) = (pred <= this->threshold).select(
                pred.constant(0), pred.constant(1));

        Tensor<TensorRank> label_rounded(label.dimensions());
        label_rounded.device(this->device) = label.round();

        Tensor<TensorRank, bool> equality(pred.dimensions());
        equality.device(this->device) = (pred_rounded == label_rounded);

        Tensor<0, int> result;
        result.device(this->device) = equality.template cast<int>().sum();

        this->correct += result(0);
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

}

#endif //FLARE_BINARY_ACCURACY_HPP
