//
// Created by R on 4/11/23.
//

#ifndef FLARE_PRECISION_HPP
#define FLARE_PRECISION_HPP

#include "metric.hpp"

namespace fl
{

// Computes the precision formula tp / (tp + fp)
template<int TensorRank>
class Precision : public Metric<TensorRank>
{
public:
    explicit Precision(Scalar threshold = 0.5) : threshold(threshold)
    {
        this->name = "precision";
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

        // precision =  tp / (tp + fp)
        Tensor<0> tp;
        tp.device(this->device) = (pred_rounded && label_rounded)
                .template cast<Scalar>().sum();

        Tensor<0> fp;
        fp.device(this->device) = (pred_rounded && label_rounded == false)
                .template cast<Scalar>().sum();

        this->true_positives = tp.coeff();
        this->false_positives = fp.coeff();
    }


    double GetMetric() const override
    {
        return true_positives / (true_positives + false_positives);
    }


    void Reset() override
    {
        this->true_positives = 0;
        this->false_positives = 0;
    }


private:
    Scalar true_positives = 0;
    Scalar false_positives = 0;

    Scalar threshold;

};

} // namespace fl

#endif //FLARE_PRECISION_HPP
