//
// Created by RKuang on 9/25/2022.
//

#ifndef ORION_ACCURACY_HPP
#define ORION_ACCURACY_HPP

#include "metric.hpp"

namespace orion
{

// initially as binary accuracy
class Accuracy : public Metric
{
public:
    explicit Accuracy(Scalar threshold);

    Scalar Compute(Sequential &model) const override;

private:
    Scalar threshold;
};

} // namespace orion

#endif //ORION_ACCURACY_HPP
