//
// Created by R on 8/7/22.
//

#ifndef FLARE_FLARE_TYPES_HPP
#define FLARE_FLARE_TYPES_HPP

#ifndef FLARE_DO_NOT_USE_THREADS
#define EIGEN_USE_THREADS
#endif

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

namespace fl
{

#ifdef FLARE_FLOAT
using Scalar = float ;
#else
using Scalar = double;
#endif

#ifndef FLARE_COLMAJOR
template<int TensorRank, typename DataType = Scalar, int StorageType = Eigen::RowMajor>
#else
template<int TensorRank, typename DataType = Scalar, int StorageType = Eigen::ColMajor>
#endif
using Tensor = Eigen::Tensor<DataType, TensorRank, StorageType>;

#ifndef FLARE_COLMAJOR
template<int TensorRank, typename DataType = Eigen::Index, int StorageType = Eigen::RowMajor>
#else
template<int TensorRank, typename DataType = Eigen::Index, int StorageType = Eigen::ColMajor>
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
namespace fl
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


#endif //FLARE_FLARE_TYPES_HPP
