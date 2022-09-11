//
// Created by macross on 8/7/22.
//

#ifndef ORION_ORION_TYPES_HPP
#define ORION_ORION_TYPES_HPP


#include <unsupported/Eigen/CXX11/Tensor>
/**
 * Differences between Eigen and Eigen::Tensor
 Notation:
    In Eigen::IndexPair<int>(f, s)
        - f = dimension of the first tensor
        - s = dimension of the second tensor
    Tensor rank 2 dimensions
        - (row [0], col [1])
    Tensor rank 3 dimensions
        - batch [0], row [1], col [2] (known as channels_first in some neural network libraries)


 Equivalent of matrix * vector operation:
    - specify the dimensions to dot product against, typically Eigen::IndexPair<int>(1, 0)

     Eigen::array<Eigen::IndexPair<int>, 1> multiply = {Eigen::IndexPair<int>(1, 0)}
     Eigen::Tensor<double, 2> left
     Eigen::Tensor<double, 2> right
     left.contract(right, multiply)


 Equivalent of matrix_transposed * vector operation:
    - e.g. tensor_left (1, 2, 3), tensor_right(2, 1)
    - Eigen::array<Eigen::IndexPair<int>, 1> multiply = {Eigen::IndexPair<int>(1, 0)}
    - while tensor_left is simply a matrix with batch=1, we perform the dot product
      between tensor_left's col-axis (1) and tensor_right's col-axis(1)
    - Eigen::Matrix equivalent is tensor_left.transpose * tensor_right

 Equivalent of cwiseProduct (elementwise product)
    - Eigen::Tensor * Eigen::Tensor
    - tensors must have same dimensions
 */

namespace orion
{

#ifdef ORION_FLOAT
typedef float Scalar;
#else
typedef double Scalar;
#endif

template<int Rank, typename DataType = Scalar>
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
