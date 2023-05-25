//
// Created by macross on 8/8/22.
//

#ifndef FLARE_SOFTMAX_HPP
#define FLARE_SOFTMAX_HPP

#include "flare/fl_types.hpp"

namespace fl
{

class Softmax
{
public:
    /**
     * Compute the activation of a tensor
     * @param tensor   Eigen::Tensor
     * @return         Eigen::Tensor or Eigen::Tensor Op
     */
    template<int TensorRank>
    static auto Activate(const Tensor <TensorRank> &tensor)
    {
        Dims <TensorRank> predict_sum_dims = tensor.dimensions();
        predict_sum_dims.back() = 1;

        Dims <TensorRank> bcast;
        bcast.fill(1);
        bcast.back() = tensor.dimension(TensorRank - 1);

        auto features_sum = tensor
                .exp()
                .sum(Dims<1>(TensorRank - 1))
                .reshape(predict_sum_dims)
                .eval()
                .broadcast(bcast);
        return tensor.exp() / features_sum;
    }


    /**
     * Compute the Jacobian of a softmax function for up to rank 4 tensors
     * @param softmax   features already activated using softmax
     * @return          Eigen Tensor
     */
    template<int TensorRank>
    static Tensor<TensorRank + 1> Gradients(const Tensor <TensorRank> &softmax)
    {
        if constexpr (TensorRank == 2) {
            return Softmax::Gradients2D(softmax);
        }
        else if constexpr(TensorRank == 3) {
            return Softmax::Gradients3D(softmax);
        }
        else if constexpr(TensorRank == 4) {
            return Softmax::Gradients4D(softmax);
        }
        else {
            throw std::invalid_argument(
                    "Softmax backpropagation supports up to rank 4 tensors");
        }
    }


private:
    static Tensor<3> Gradients2D(const Tensor<2> &softmax)
    {
        // create a rank 3 tensor, [N,row,col] where the last 2 dims are the
        // Jacobian matrices of each sample of the rank 2 batch tensor
        Tensor<3> softmax_grad(softmax.dimension(0),
                               softmax.dimension(1),
                               softmax.dimension(1));

        for (Eigen::Index batch = 0; batch < softmax.dimension(0); batch++) {
            // begin calculating the Jacobian matrix of each row
            // according to the formula, where S is the softmax gradient Jacobian
            // S_i * (1 - S_i)   for i == j
            // -S_i * S_j        for i != j
            // or compactly as S_i * (d_ij - S_j), where d_ij = Kronecker delta (fancy ternary)
            for (Eigen::Index row = 0; row < softmax.dimension(1); row++) {
                for (Eigen::Index col = 0; col < softmax.dimension(1); col++) {
                    if (row != col) {
                        softmax_grad(batch, row, col) =
                                -softmax(batch, row) * softmax(batch, col);
                    }
                    else {
                        softmax_grad(batch, row, col) =
                                softmax(batch, row) * (1 - softmax(batch, row));
                    }
                }
            }
        }

        return softmax_grad;
    }


    // Only difference from Gradients2D is an extra loop between the batch
    // and last 2 dimensions. If implementing higher rank gradients, just add
    // another for loop if a better solution isn't found``
    static Tensor<4> Gradients3D(const Tensor<3> &softmax)
    {
        // create a rank 3 tensor, [N,row,col] where the last 2 dims are the
        // Jacobian matrices of each sample of the rank 2 batch tensor
        Tensor<4> softmax_grad(softmax.dimension(0),
                               softmax.dimension(1),
                               softmax.dimension(2),
                               softmax.dimension(2));

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) default(none) shared(softmax, softmax_grad)
#endif
        for (Eigen::Index a = 0; a < softmax.dimension(0); a++) {
            for (Eigen::Index b = 0; b < softmax.dimension(1); b++) {
                // begin calculating the Jacobian matrix of each row
                // according to the formula, where S is the softmax gradient Jacobian
                // S_i * (1 - S_i)   for i == j
                // -S_i * S_j        for i != j
                // or compactly as S_i * (d_ij - S_j), where d_ij = Kronecker delta (fancy ternary)
                for (Eigen::Index row = 0; row < softmax.dimension(2); row++) {
                    for (Eigen::Index col = 0; col < softmax.dimension(2); col++) {
                        if (row != col) {
                            softmax_grad(a, b, row, col) =
                                    -softmax(a, b, row) * softmax(a, b, col);
                        }
                        else {
                            softmax_grad(a, b, row, col) =
                                    softmax(a, b, row) * (1 - softmax(a, b, row));
                        }
                    }
                }
            }
        }

        return softmax_grad;
    }


    static Tensor<5> Gradients4D(const Tensor<4> &softmax)
    {
        // create a rank 3 tensor, [N,row,col] where the last 2 dims are the
        // Jacobian matrices of each sample of the rank 2 batch tensor
        Tensor<5> softmax_grad(softmax.dimension(0),
                               softmax.dimension(1),
                               softmax.dimension(2),
                               softmax.dimension(3),
                               softmax.dimension(3));

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) default(none) shared(softmax, softmax_grad)
#endif
        for (Eigen::Index a = 0; a < softmax.dimension(0); a++) {
            for (Eigen::Index b = 0; b < softmax.dimension(1); b++) {
                for (Eigen::Index c = 0; c < softmax.dimension(2); c++) {
                    // begin calculating the Jacobian matrix of each row
                    // according to the formula, where S is the softmax gradient Jacobian
                    // S_i * (1 - S_i)   for i == j
                    // -S_i * S_j        for i != j
                    // or compactly as S_i * (d_ij - S_j), where d_ij = Kronecker delta (fancy ternary)
                    for (Eigen::Index row = 0; row < softmax.dimension(3); row++) {
                        for (Eigen::Index col = 0;
                             col < softmax.dimension(3); col++) {
                            if (row != col) {
                                softmax_grad(a, b, c, row, col) =
                                        -softmax(a, b, c, row) *
                                        softmax(a, b, c, col);
                            }
                            else {
                                softmax_grad(a, b, c, row, col) =
                                        softmax(a, b, c, row) *
                                        (1 - softmax(a, b, c, row));
                            }
                        }
                    }
                }
            }
        }

        return softmax_grad;
    }
};

} // namespace fl

#endif //FLARE_SOFTMAX_HPP
