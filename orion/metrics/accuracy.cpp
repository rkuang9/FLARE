//
// Created by RKuang on 9/25/2022.
//

#include "accuracy.hpp"

namespace orion
{

Accuracy::Accuracy(Scalar threshold) : threshold(threshold)
{
    // nothing to do here
}


Scalar Accuracy::Compute(Sequential &model) const
{
    throw std::invalid_argument("Accuracy metrics not implemented");
}

}