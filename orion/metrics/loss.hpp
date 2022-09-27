//
// Created by RKuang on 9/25/2022.
//

#ifndef ORION_LOSS_HPP
#define ORION_LOSS_HPP

#include "metric.hpp"

namespace orion
{

class Loss : public Metric
{
public:
    Loss();

    Scalar Compute(Sequential &model) const override;
};

}

#endif //ORION_LOSS_HPP
