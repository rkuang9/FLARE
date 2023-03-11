//
// Created by macross on 8/8/22.
//

#ifndef ORION_SOFTMAX_HPP
#define ORION_SOFTMAX_HPP

#include "orion/orion_types.hpp"

namespace orion
{

class Softmax
{
public:
    /**
     * @param dimension   dimension on which to sum along
     */
    template<int TensorRank>
    static auto Activate(const Tensor<TensorRank> &features,
                                       int dimension = TensorRank - 1)
    {
        Dims<TensorRank> predict_sum_dims = features.dimensions();
        predict_sum_dims.back() = 1;

        Dims<TensorRank> bcast;
        bcast.fill(1);
        bcast.back() = features.dimension(dimension);

        auto features_sum = features
                .exp()
                .sum(Dims<1>(dimension))
                .reshape(predict_sum_dims)
                .eval()
                .broadcast(bcast);
        return features.exp() / features_sum;
    }

// https://github.com/tensorflow/tensorflow/blob/9a4af32dae70849e8175c17b68f8627e926d28e4/tensorflow/core/kernels/sparse/kernels_gpu.cu.cc
    /**
     * Compute the Jacobian of a softmax function
     * @tparam TensorRank
     * @param softmax   features already activated using softmax
     * @return
     */
    template<int TensorRank>
    static Tensor<TensorRank + 1> Gradients(const Tensor<TensorRank> &softmax)
    {
        if constexpr (TensorRank == 2) {
            return Softmax::Gradients2D(softmax);
        }
        else {
            throw std::invalid_argument(
                    "Softmax backpropagation currently only supported for rank 2 tensors");
        }
    }


private:
    static Tensor<3> Gradients2D(const Tensor<2> &softmax)
    {
        // create a rank 3 tensor, [N,row,col] where the last 2 dims are the
        // Jacobian matrices of each sample of the rank 2 batch tensor
        Tensor<3> softmax_grad(softmax.dimension(0), softmax.dimension(1),
                               softmax.dimension(1));

        for (int batch = 0; batch < softmax.dimension(0); batch++) {
            // begin calculating the Jacobian matrix of each row
            // according to the formula, where S is the softmax gradient Jacobian
            // S_i * (1 - S_i)   for i == j
            // -S_i * S_j        for i != j
            // or compactly as S_i * (d_ij - S_j), where d_ij = Kronecker delta (fancy ternary)
            for (int row = 0; row < softmax.dimension(1); row++) {
                for (int col = 0; col < softmax.dimension(1); col++) {
                    if (row == col) {
                        softmax_grad(batch, row, col) =
                                softmax(batch, row) * (1 - softmax(batch, row));
                    }
                    else {
                        softmax_grad(batch, row, col) =
                                -softmax(batch, row) * softmax(batch, col);
                    }
                }
            }
        }

        return softmax_grad;
    }
};

} // namespace orion

#endif //ORION_SOFTMAX_HPP
