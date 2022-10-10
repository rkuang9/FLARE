//
// Created by macross on 10/10/22.
//

#include "maxpooling2d.hpp"

namespace orion
{
template<int PoolRank>
void MaxPooling2D<PoolRank>::Forward(const Tensor<2> &inputs)
{
    Layer::Forward(inputs);
}


template<int PoolRank>
void MaxPooling2D<PoolRank>::Forward(const Tensor<3> &inputs)
{
    Layer::Forward(inputs);
}


template<int PoolRank>
void MaxPooling2D<PoolRank>::Forward(const Tensor<4> &inputs)
{
    Layer::Forward(inputs);
}


template<int PoolRank>
void MaxPooling2D<PoolRank>::Backward(const LossFunction &loss_function)
{
    Layer::Backward(loss_function);
}


template<int PoolRank>
void MaxPooling2D<PoolRank>::Forward(const Layer &prev)
{
    Layer::Forward(prev);
}


template<int PoolRank>
void MaxPooling2D<PoolRank>::Backward(const Layer &next)
{
    Layer::Backward(next);
}


template<int PoolRank>
void MaxPooling2D<PoolRank>::Update(Optimizer &optimizer)
{
    Layer::Update(optimizer);
}


template<int PoolRank>
const Tensor<2> &MaxPooling2D<PoolRank>::GetOutput2D() const
{
    return Layer::GetOutput2D();
}


template<int PoolRank>
const Tensor<3> &MaxPooling2D<PoolRank>::GetOutput3D() const
{
    return Layer::GetOutput3D();
}


template<int PoolRank>
const Tensor<4> &MaxPooling2D<PoolRank>::GetOutput4D() const
{
    return Layer::GetOutput4D();
}


template<int PoolRank>
const Tensor<2> &MaxPooling2D<PoolRank>::GetInputGradients2D() const
{
    return Layer::GetInputGradients2D();
}


template<int PoolRank>
const Tensor<3> &MaxPooling2D<PoolRank>::GetInputGradients3D() const
{
    return Layer::GetInputGradients3D();
}


template<int PoolRank>
const Tensor<4> &MaxPooling2D<PoolRank>::GetInputGradients4D() const
{
    return Layer::GetInputGradients4D();
}


template<int PoolRank>
const Tensor<2> &MaxPooling2D<PoolRank>::GetWeights() const
{
    return Layer::GetWeights();
}


template<int PoolRank>
const Tensor<4> &MaxPooling2D<PoolRank>::GetWeights4D() const
{
    return Layer::GetWeights4D();
}


template<int PoolRank>
const Tensor<2> &MaxPooling2D<PoolRank>::GetWeightGradients() const
{
    return Layer::GetWeightGradients();
}


template<int PoolRank>
const Tensor<4> &MaxPooling2D<PoolRank>::GetWeightGradients4D() const
{
    return Layer::GetWeightGradients4D();
}


template<int PoolRank>
void MaxPooling2D<PoolRank>::SetWeights(const Tensor<2> &weights)
{
    Layer::SetWeights(weights);
}


template<int PoolRank>
void MaxPooling2D<PoolRank>::SetWeights(const Tensor<4> &weights)
{
    Layer::SetWeights(weights);
}


template<int PoolRank>
const Tensor<2> &MaxPooling2D<PoolRank>::GetBias() const
{
    return Layer::GetBias();
}


template<int PoolRank>
void MaxPooling2D<PoolRank>::SetBias(const Tensor<2> &bias)
{
    Layer::SetBias(bias);
}


template<int PoolRank>
void MaxPooling2D<PoolRank>::SetBias(const Tensor<4> &bias)
{
    Layer::SetBias(bias);
}


template<int PoolRank>
int MaxPooling2D<PoolRank>::GetInputRank() const
{
    return 0;
}


template<int PoolRank>
int MaxPooling2D<PoolRank>::GetOutputRank() const
{
    return 0;
}
} // orion