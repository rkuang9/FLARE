//
// Created by macross on 8/7/22.
//

#ifndef ORION_RANDOM_GENERATOR_HPP
#define ORION_RANDOM_GENERATOR_HPP

#include "orion/orion_types.hpp"
#include <random>

namespace orion::random
{

Tensor2D RandomUniform(int rows, int cols, Scalar min, Scalar max);

Tensor2D RandomNormal(int rows, int cols, Scalar mean, Scalar stddev);

} // namespace orion

#endif //ORION_RANDOM_GENERATOR_HPP
