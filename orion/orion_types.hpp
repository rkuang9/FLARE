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
 * For convolutional layers: (batch, rows, cols, channels)
 * For embedding layers: (batch, rows, cols)
 * For dense layers: (rows, batch)
 */

namespace orion
{

#ifdef ORION_FLOAT
using Scalar = float ;
#else
using Scalar = double;
#endif

#ifdef ORION_COLMAJOR
template<int InputTensorRank, typename DataType = Scalar, int StorageType = Eigen::ColMajor>
#else
template<int TensorRank, typename DataType = Scalar, int StorageType = Eigen::RowMajor>
#endif
using Tensor = Eigen::Tensor<DataType, TensorRank, StorageType>;

#ifdef ORION_COLMAJOR
template<int InputTensorRank, typename DataType = Eigen::Index, int StorageType = Eigen::ColMajor>
#else
template<int TensorRank, typename DataType = Eigen::Index, int StorageType = Eigen::RowMajor>
#endif
using Dims = typename Eigen::DSizes<DataType, TensorRank>;


// Maps, create tensors on existing data
template<int TensorRank>
using TensorMap = Eigen::TensorMap<Tensor<TensorRank>>;

template<int TensorRank>
using TensorMapConst = Eigen::TensorMap<const Tensor<TensorRank>>;

// contraction dimensions
using ContractDim = Eigen::array<Eigen::IndexPair<int>, 1>;
using Axes = Eigen::IndexPair<int>;

}


// convolution, pooling typedefs and classes
namespace orion
{

using Input = Dims<3>;
using Kernel = Dims<2>;
using Stride = Dims<2>;
using Dilation = Dims<2>;
using Inflate = Dims<2>;
using Padding = Eigen::PaddingType;

// pooling typedefs
using PoolSize = Dims<2>;
}


#endif //ORION_ORION_TYPES_HPP
