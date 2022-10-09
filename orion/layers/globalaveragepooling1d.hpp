//
// Created by RKuang on 9/17/2022.
//

#ifndef ORION_GLOBALAVERAGEPOOLING1D_H
#define ORION_GLOBALAVERAGEPOOLING1D_H

#include "layer.hpp"

namespace orion
{

class GlobalAveragePooling1D : public Layer
{

public:

    /**
     * Create a layer that computes the average of a
     * tensor over the column dimension
     */
    GlobalAveragePooling1D();

    ~GlobalAveragePooling1D() override = default;

    void Forward(const Tensor<3> &inputs) override;

    void Forward(const Layer &prev) override;

    void Backward(const Layer &next) override;

    void Backward(const LossFunction &loss_function) override;

    const Tensor<2> &GetOutput2D() const override;

    const Tensor<3> &GetInputGradients3D() const override;

    const Tensor<2> &GetWeights() const override;

    int GetInputRank() const override;

    int GetOutputRank() const override;

public:
    void Backward(const Tensor<2> &gradients);

    Tensor<3> X; // layer input, TODO: layer probably doesn't need to store this
    Tensor<2> Z; // layer output
    Tensor<3> dL_dZ;

    const Tensor<2> w; // empty weights, exists so that gradient check can skip over it

    // dimensions for a col-wise mean
    Eigen::array<Eigen::Index, 1> avg_over_dim{1};
};

}

#endif //ORION_GLOBALAVERAGEPOOLING1D_H
