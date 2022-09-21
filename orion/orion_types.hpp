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

template<int Rank, typename DataType = Scalar, int = Eigen::ColMajor>
using Tensor = Eigen::Tensor<DataType, Rank, Eigen::ColMajor>;

typedef Eigen::Tensor<Scalar, 4, Eigen::ColMajor> Tensor4D;
typedef Eigen::Tensor<Scalar, 3, Eigen::ColMajor> Tensor3D;
typedef Eigen::Tensor<Scalar, 2, Eigen::ColMajor> Tensor2D;
typedef Eigen::Tensor<Scalar, 1, Eigen::ColMajor> Tensor1D;



// a different view on an existing Tensor, doesn't copy
template<int Rank>
using TensorMap = Eigen::TensorMap<Tensor<Rank>>;

template<int Rank>
using TensorMapConst = Eigen::TensorMap<const Tensor<Rank>>;


typedef Eigen::TensorMap<Tensor4D> Tensor4DMap;
typedef Eigen::TensorMap<Tensor3D> Tensor3DMap;
typedef Eigen::TensorMap<Tensor2D> Tensor2DMap;
typedef Eigen::TensorMap<Tensor1D> Tensor1DMap;


typedef Eigen::array<Eigen::IndexPair<int>, 1> ContractDim;
typedef Eigen::IndexPair<int> Axes;

}


#endif //ORION_ORION_TYPES_HPP
