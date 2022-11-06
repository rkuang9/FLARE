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
template<int TensorRank, typename DataType = Scalar, int StorageType = Eigen::ColMajor>
#else
template<int TensorRank, typename DataType = Scalar, int StorageType = Eigen::RowMajor>
#endif
using Tensor = Eigen::Tensor<DataType, TensorRank, StorageType>;

#ifdef ORION_COLMAJOR
template<int TensorRank, typename DataType = Eigen::Index, int StorageType = Eigen::ColMajor>
#else
template<int TensorRank, typename DataType = Eigen::Index, int StorageType = Eigen::RowMajor>
#endif
//using Dims = typename Eigen::Tensor<Eigen::Index, TensorRank, StorageType>::Dimensions;
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

// this class was made so that Clion displays shorter arguments rather than the
// firstDimension and secondDimension that clog up the screen width
class Input : public Dims<3>
{
public:
    Input(Eigen::Index height, Eigen::Index width, Eigen::Index channels)
            : Tensor<3>::Dimensions(height, width, channels) {}

    Eigen::Index height() const { return this->at(0); }
    Eigen::Index width() const { return this->at(1); }
    Eigen::Index channels() const { return this->at(2); }
};

// same deal with class Input
class _dims2D: public Dims<2>
{
public:
    _dims2D(Eigen::Index height, Eigen::Index width) : Dims<2>(height, width) {}

    Eigen::Index height() const { return this->at(0); }
    Eigen::Index width() const { return this->at(1); }
};

using Kernel = _dims2D;
using Stride = _dims2D;
using Dilation = _dims2D;
using Padding = Eigen::PaddingType;

// pooling typedefs
using PoolSize = _dims2D;
}


#endif //ORION_ORION_TYPES_HPP
