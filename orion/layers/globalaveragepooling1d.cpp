//
// Created by RKuang on 9/17/2022.
//

#include "globalaveragepooling1d.hpp"

namespace orion
{

GlobalAveragePooling1D::GlobalAveragePooling1D() = default;


void GlobalAveragePooling1D::Forward(const Tensor<3> &inputs)
{
    this->X = inputs;
    Tensor<2>::Dimensions(1, 1);
    // compute mean col-wise
    this->Z = inputs.mean(this->avg_over_dim);
}


void GlobalAveragePooling1D::Forward(const Layer &prev)
{
    this->Forward(prev.GetOutput3D());
}


void GlobalAveragePooling1D::Backward(Layer &next)
{
    // next.weights.transpose * next.dL_dZ
    this->Backward(next.GetWeights().contract(
            next.GetInputGradients2D(), ContractDim{Axes{0, 0}}));
}


void GlobalAveragePooling1D::Backward(const LossFunction &loss_function)
{
    // pass the dL / dA term from dL / dZ = (dL / dA) * (dA / dZ)
    // divide by num outputs
    this->Backward(loss_function.GetGradients2D());
}


void GlobalAveragePooling1D::Backward(const Tensor<2> &gradients)
{
    // dL / dZ = (dL / dA) * (dA / dZ) = loss gradients * activation gradients
    // where dA / dZ = d((x1 + x2 + ... + x_n) / n) / dZ = 1 / n = input.rows
    Tensor<2> grad =
            gradients / (Scalar) this->X.dimension(this->avg_over_dim.front());

    this->dL_dZ.resize(this->X.dimensions());

    // reshape dL_dZ into input tensor's shape, each row belongs to a batch
    for (Eigen::Index row = 0; row < grad.dimension(0); row++) {
        // reshape each gradient row into the dimensions of one batch
        this->dL_dZ.chip(row, 0) = grad.chip(row, 0) // 0 = row dimension
                .reshape(Tensor<2>::Dimensions(1, grad.dimension(1)))
                .eval()
                .broadcast(Tensor<2>::Dimensions(this->X.dimension(1), 1));
    }

    orion_assert(this->dL_dZ.dimensions() == this->X.dimensions(), this->name <<
            " GlobalAveragePooling1D::Backward loss gradients dimensions " <<
            this->dL_dZ.dimensions() << " do not match input tensor dimensions " <<
            this->X.dimensions());
}


const Tensor<2> &GlobalAveragePooling1D::GetOutput2D() const
{
    return this->Z;
}


Tensor<3> GlobalAveragePooling1D::GetInputGradients3D()
{
    return this->dL_dZ; // TODO: re-evaluate dL_dX calculation
}


const Tensor<2> &GlobalAveragePooling1D::GetWeights() const
{
    return this->w;
}


int GlobalAveragePooling1D::GetInputRank() const
{
    return 3;
}


int GlobalAveragePooling1D::GetOutputRank() const
{
    return 2;
}

}