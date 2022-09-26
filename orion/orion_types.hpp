//
// Created by macross on 8/7/22.
//

#ifndef ORION_ORION_TYPES_HPP
#define ORION_ORION_TYPES_HPP


#include <unsupported/Eigen/CXX11/Tensor>

/**
 * Tensor dimension convention
 *
 * Rank 2: (rows, cols)
 * Rank 3: (batch, rows, cols)
 * Rank 4: (batch, rows, cols, channels)
 *
 * For convolutional layers: (rows, cols, channels)
 * For embedding layers: (batch, rows, cols)
 * For dense layers: (rows, cols)
 */

namespace orion
{

#ifdef ORION_FLOAT
typedef float Scalar;
#else
typedef double Scalar;
#endif


template<int Rank, typename DataType = Scalar, int Storage = Eigen::ColMajor>
using Tensor = Eigen::Tensor<DataType, Rank, Storage>;


// a different view on an existing Tensor, doesn't copy
template<int Rank>
using TensorMap = Eigen::TensorMap<Tensor<Rank>>;


template<int Rank>
using TensorMapConst = Eigen::TensorMap<const Tensor<Rank>>;


typedef Eigen::array<Eigen::IndexPair<int>, 1> ContractDim;
typedef Eigen::IndexPair<int> Axes;

}


#endif //ORION_ORION_TYPES_HPP
