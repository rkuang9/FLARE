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

#ifdef ORION_COLMAJOR
template<int TensorRank, typename DataType = Scalar, int StorageType = Eigen::ColMajor>
#else
template<int TensorRank, typename DataType = Scalar, int StorageType = Eigen::RowMajor>
#endif
using Tensor = Eigen::Tensor<DataType, TensorRank, StorageType>;

#ifdef ORION_COLMAJOR
template<int TensorRank, typename DataType = Scalar, int StorageType = Eigen::ColMajor>
#else
template<int TensorRank, typename DataType = Scalar, int StorageType = Eigen::RowMajor>
#endif
using Dims = typename Eigen::Tensor<DataType, TensorRank, StorageType>::Dimensions;


// Maps, create tensors on existing data
template<int TensorRank>
using TensorMap = Eigen::TensorMap<Tensor<TensorRank>>;

template<int TensorRank>
using TensorMapConst = Eigen::TensorMap<const Tensor<TensorRank>>;

// contraction dimensions
typedef Eigen::array<Eigen::IndexPair<int>, 1> ContractDim;
typedef Eigen::IndexPair<int> Axes;

// convolution typedefs
typedef Tensor<3>::Dimensions Input;
typedef Tensor<2>::Dimensions Kernel;
typedef Tensor<2>::Dimensions Stride;
typedef Tensor<2>::Dimensions Dilation;
using Padding = Eigen::PaddingType;

// pooling typedefs


}


#endif //ORION_ORION_TYPES_HPP
