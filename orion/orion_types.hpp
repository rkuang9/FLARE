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
template<int Rank, typename DataType = Scalar, int StorageType = Eigen::ColMajor>
#else
template<int Rank, typename DataType = Scalar, int StorageType = Eigen::RowMajor>
#endif
using Tensor = Eigen::Tensor<DataType, Rank, StorageType>;

#ifdef ORION_COLMAJOR
template<int Rank, typename DataType = Scalar, int StorageType = Eigen::ColMajor>
#else
template<int Rank, typename DataType = Scalar, int StorageType = Eigen::RowMajor>
#endif
using Dims = typename Eigen::Tensor<DataType, Rank, StorageType>::Dimensions;


// Maps, create tensors on existing data
template<int Rank>
using TensorMap = Eigen::TensorMap<Tensor<Rank>>;

template<int Rank>
using TensorMapConst = Eigen::TensorMap<const Tensor<Rank>>;

// contraction dimensions
typedef Eigen::array<Eigen::IndexPair<int>, 1> ContractDim;
typedef Eigen::IndexPair<int> Axes;


// convolution typedefs
typedef Eigen::PaddingType PaddingType;
typedef Tensor<2>::Dimensions Stride;
typedef Tensor<2>::Dimensions Kernel;
typedef Tensor<2>::Dimensions Dilation;

}


#endif //ORION_ORION_TYPES_HPP
