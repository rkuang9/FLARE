//
// Created by R on 4/12/23.
//

#ifndef FLARE_RECALL_HPP
#define FLARE_RECALL_HPP

#include "metric.hpp"

namespace fl
{

// Computes the precision formula tp / (tp + fn)
template<int TensorRank>
class Recall: public Metric<TensorRank>
{
public:
    public:
    explicit Recall(Scalar threshold = 0.5) : threshold(threshold)
    {
        this->name = "recall";
    }


    void operator()(const Tensor <TensorRank> &pred,
                    const Tensor <TensorRank> &label) override
    {
        FL_REQUIRES(
                pred.dimensions() == label.dimensions(),
                this->name << " pred dimensions " << pred.dimensions()

                           << " != label dimensions " << label.dimensions());

        Tensor<TensorRank> pred_rounded(pred.dimensions());
        pred_rounded.device(this->device) = (pred <= this->threshold).select(
                pred.constant(0), pred.constant(1));

        Tensor<TensorRank> label_rounded(label.dimensions());
        label_rounded.device(this->device) = label.round();

        // recall =  tp / (tp + fn)
        Tensor<0> tp;
        tp.device(this->device) = (pred_rounded && label_rounded)
                .template cast<Scalar>().sum();

        Tensor<0> fn; // predicted false, but was true
        fn.device(this->device) = (pred_rounded == false && label_rounded)
                .template cast<Scalar>().sum();

        this->true_positives = tp.coeff();
        this->false_negatives = fn.coeff();
    }


    double GetMetric() const override
    {
        return true_positives / (true_positives + false_negatives);
    }


    void Reset() override
    {
        this->true_positives = 0;
        this->false_negatives = 0;
    }


private:
    Scalar true_positives = 0;
    Scalar false_negatives = 0;

    Scalar threshold;

};

} // namespace fl

#endif //FLARE_RECALL_HPP
