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
     * tensor over a given dimension
     *
     * @param dimension   the dimension to average over, defaults col-wise
     */
    explicit GlobalAveragePooling1D(Eigen::Index dimension = 1);

    ~GlobalAveragePooling1D() override = default;

    void Forward(const Tensor<3> &inputs) override;

    void Forward(const Layer &prev) override;

    void Backward(const Layer &next) override;

    const Tensor<2> &GetOutput2D() const override;

    const Tensor<3> &GetOutput3D() const override;

    const Tensor<2> &GetInputGradients2D() const override;

    int GetInputRank() const override;

    int GetOutputRank() const override;

private:
    Eigen::Index target_dim; // dimension to average over

};

}

#endif //ORION_GLOBALAVERAGEPOOLING1D_H
