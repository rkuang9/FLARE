//
// Created by RKuang on 9/17/2022.
//

#include "globalaveragepooling1d.hpp"

namespace orion
{

GlobalAveragePooling1D::GlobalAveragePooling1D(Eigen::Index dimension) :
        avg_over_dim(Eigen::array<Eigen::Index, 1>{dimension})
{
}


void GlobalAveragePooling1D::Forward(const Tensor<3> &inputs)
{
    this->Z = inputs.mean(this->avg_over_dim);
}


void GlobalAveragePooling1D::Forward(const Layer &prev)
{
    this->Forward(prev.GetOutput3D());
}


void GlobalAveragePooling1D::Backward(const Layer &next)
{
    // next.weights.transpose * next.dL_dZ
    this->Backward(next.GetWeights().contract(
            next.GetInputGradients2D(), ContractDim{Axes{0, 0}}));
}


void GlobalAveragePooling1D::Backward(const Loss &loss_function)
{
    // pass the dL / dA term from dL / dZ = (dL / dA) * (dA / dZ)
    this->Backward(loss_function.GetGradients2D());
}


void GlobalAveragePooling1D::Backward(const Tensor<2> &gradients)
{
    // dL / dZ = (dL / dA) * (dA / dZ) = gradients * activation gradients
    // dA / dZ = d((x1 + x2 + ... + x_n) / n) / dZ = 1 / n
    this->dL_dZ = gradients * (Scalar)this->X.dimension(this->avg_over_dim.front());
}


const Tensor<2> &GlobalAveragePooling1D::GetOutput2D() const
{
    return this->Z;
}


const Tensor<2> &GlobalAveragePooling1D::GetInputGradients2D() const
{
    return this->dL_dZ;
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