//
// Created by RKuang on 9/25/2022.
//

#ifndef ORION_METRIC_HPP
#define ORION_METRIC_HPP

#include <string>
#include "orion/orion_types.hpp"


namespace orion
{

// forward declaration
class Sequential;

class Metric
{
public:
    virtual Scalar Compute(Sequential &model) const = 0;

    std::string name = "metric";
};

} // namespace orion

#endif //ORION_METRIC_HPP
