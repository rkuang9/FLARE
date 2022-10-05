//
// Created by macross on 9/1/22.
//

#include "conv2d.hpp"

namespace orion
{


template<typename Activation>
Conv2D<Activation>::Conv2D(int filters, int kernel_height, int kernel_width,
                           const Tensor<2>::Dimensions &input_dims,
                           const Initializer<2> &initializer)
{
#ifndef ORION_ROWMAJOR
    throw std::invalid_argument(
            "Conv2D only supports the NHWC tensor format, use ORION_ROWMAJOR instead");
#endif
}

// !!! TODO: Sequential should swap H&W and reswap back to original when done
// https://stackoverflow.com/questions/55532819/replicating-tensorflows-conv2d-operating-using-eigen-tensors
template<typename Activation>
void Conv2D<Activation>::Forward(const Tensor<4> &inputs)
{
    // in row-major Eigen uses format NWHC, to swap the dimensions W and H
    // we can use shuffle(Tensor<4>::Dimensions(W_dim, H_dim))
    this->X = inputs;

    // convert input NHWC format to Eigen expected NWHC
    auto nwhc = inputs.shuffle(this->shuffle_hw).reverse(this->reverse_hw);


}


template<typename Activation>
void Conv2D<Activation>::Backward(const LossFunction &loss_function)
{
    // https://www.youtube.com/watch?v=Pn7RK7tofPg
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


template<typename Activation>
auto Conv2D<Activation>::NHWCToNWHC(const Tensor<4> &tensor)
{
    // swap height and width dimensions by transpose, then reverse cols
    return tensor.shuffle(Tensor<4>::Dimensions(0, 2, 1, 3))
            .reverse(Tensor<4, bool>::Dimensions(false, false, true, false));
}


template<typename Activation>
auto Conv2D<Activation>::NWHCToNHWC(const Tensor<4> &tensor)
{
    // swap width and height dimensions by transpose, then reverse rows
    return tensor.shuffle(Tensor<4>::Dimensions(0, 2, 1, 3))
            .reverse(Tensor<4, bool>::Dimensions(false, true, false, false));
}

}