//
// Created by macross on 8/7/22.
//

#ifndef ORION_WEIGHTS_HPP
#define ORION_WEIGHTS_HPP

#include "orion/orion_types.hpp"
#include "random_generator.hpp"

namespace orion
{

class Initializer {
public:
    virtual Tensor2D Initialize(int rows, int cols) const = 0;
};


class GlorotUniform: public Initializer
{
    /**
     * Initialize a matrix with shape (rows, cols) on a uniform distribution
     *
     * @param rows number of rows, also the number of layer output units
     * @param cols number of cols, also the number of layer input units
     * @return
     */
    Tensor2D Initialize(int rows, int cols) const override
    {
        Scalar limit = std::sqrt(6.0 / (cols + rows));
        return random::RandomUniform(rows, cols, -limit, limit);
    }
};


class GlorotNormal: public Initializer
{
    /**
     * Initialize a matrix with shape (rows, cols) on a normal distribution
     *
     * @param rows number of rows, also the number of layer output units
     * @param cols number of cols, also the number of layer input units
     * @return
     */
    Tensor2D Initialize(int rows, int cols) const override
    {
        Scalar stddev = std::sqrt(2 / (cols + rows));
        return random::RandomNormal(rows, cols, 0, stddev);
    }
};

using XavierUniform = GlorotUniform; // also known as Xavier
using XavierNormal = GlorotNormal;


class HeUniform: public Initializer
{
    Tensor2D Initialize(int rows, int cols) const override
    {
        Scalar limit = std::sqrt(6.0 / cols);
        return random::RandomUniform(rows, cols, -limit, limit);
    }
};


class HeNormal: public Initializer
{
public:
    Tensor2D Initialize(int rows, int cols) const override
    {
        Scalar stddev = std::sqrt(2.0 / cols);
        return random::RandomNormal(rows, cols, 0, stddev);
    }
};


class LecunUniform: public Initializer
{
    Tensor2D Initialize(int rows, int cols) const override
    {
        Scalar limit = std::sqrt(3.0 / cols);
        return random::RandomUniform(rows, cols, -limit, limit);
    }
};


class LecunNormal: public Initializer
{
    Tensor2D Initialize(int rows, int cols) const override
    {
        Scalar stddev = std::sqrt(1.0 / cols);
        return random::RandomNormal(rows, cols, 0, stddev);
    }
};

} // namespace orion

#endif //ORION_WEIGHTS_HPP
