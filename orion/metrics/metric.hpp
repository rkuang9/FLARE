//
// Created by RKuang on 9/25/2022.
//

#ifndef ORION_METRIC_HPP
#define ORION_METRIC_HPP

#include <string>
#include "orion/orion_types.hpp"
#include "orion/sequential.hpp"

namespace orion
{

class Metric
{
public:
    virtual Scalar Compute(const Sequential &model) const = 0;
    std::string name = "callback";
};

} // namespace orion

#endif //ORION_METRIC_HPP
