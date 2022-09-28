//
// Created by macross on 9/1/22.
//

#include "conv2d.hpp"

namespace orion
{


template<typename Activation>
Conv2D<Activation>::Conv2D(int filters, int kernel_height, int kernel_width,
                           const Tensor<2>::Dimensions &input_dims,
                           const Initializer &initializer)
        : filters(initializer.Initialize(
        Tensor<3>::Dimensions(filters, kernel_height, kernel_width),
        input_dims[0], input_dims[1]))
{

}


template<typename Activation>
void Conv2D<Activation>::Forward(const Tensor<4> &inputs)
{
    this->X = inputs;
}


template<typename Activation>
void Conv2D<Activation>::Backward(const LossFunction &loss_function)
{
    Layer::Backward(loss_function);
}


template<typename Activation>
void Conv2D<Activation>::Forward(const Layer &prev)
{
    Layer::Forward(prev);
}


template<typename Activation>
void Conv2D<Activation>::Backward(const Layer &next)
{
    Layer::Backward(next);
}


template<typename Activation>
void Conv2D<Activation>::Update(Optimizer &optimizer)
{
    Layer::Update(optimizer);
}


template<typename Activation>
const Tensor<4> &Conv2D<Activation>::GetOutput4D() const
{
    return Layer::GetOutput2D();
}


template<typename Activation>
const Tensor<4> &Conv2D<Activation>::GetInputGradients4D() const
{
    return Layer::GetInputGradients2D();
}


template<typename Activation>
const Tensor<3> &Conv2D<Activation>::GetWeightGradients3D() const
{
    return Layer::GetWeightGradients();
}


template<typename Activation>
void Conv2D<Activation>::SetWeights(const Tensor<3> &weights)
{

}


template<typename Activation>
void Conv2D<Activation>::SetBias(const Tensor<3> &bias)
{
    Layer::SetBias(bias);
}


template<typename Activation>
int Conv2D<Activation>::GetInputRank() const
{
    return 4;
}


template<typename Activation>
int Conv2D<Activation>::GetOutputRank() const
{
    return 4;
}

}