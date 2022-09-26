//
// Created by RKuang on 9/25/2022.
//

#ifndef ORION_ACCURACY_HPP
#define ORION_ACCURACY_HPP

#include "metric.hpp"

namespace orion
{

// binary accuracy initially
class Accuracy : public Metric
{
public:
    Accuracy(Scalar threshold);

    void Compute(const Sequential &model) const override;

private:
    Scalar threshold;
};

} // namespace orion

#endif //ORION_ACCURACY_HPP
