//
// Created by macross on 9/1/22.
//

#include "conv2d.hpp"

namespace orion
{


template<typename Activation>
Conv2D<Activation>::Conv2D(int filters, Kernel kernel, Stride stride,
                           Dilation dilation, PaddingType padding,
                           const Initializer<2> &initializer) :
        kernel(kernel),
        stride(stride),
        dilation(dilation),
        padding(padding)
{
#ifdef ORION_COLMAJOR
    throw std::invalid_argument(
            "Conv2D only supports the NHWC tensor format, use ORION_ROWMAJOR instead");
#endif

    orion_assert(kernel.size() == 2 && stride.size() == 2 && dilation.size() == 2,
                 "kernel, stride, and dilation dimensions 2 values");

    if (padding == Eigen::PaddingType::PADDING_SAME) {
        this->padding_h = (kernel[0] - 1) / 2;
        this->padding_w = (kernel[0] - 1) / 2;
    }
}


// https://stackoverflow.com/questions/55532819/replicating-tensorflows-conv2d-operating-using-eigen-tensors
template<typename Activation>
void Conv2D<Activation>::Forward(const Tensor<4> &inputs)
{
    // if ROW_MAJOR, inputs expected to be NHWC
    this->X = inputs;

    // inputs should have dimensions NHWC (batch, height, width, channels, technically NWHC)
    auto batches = inputs.dimension(0);
    auto output_w = (inputs.dimension(1) + 2 * this->padding_w - kernel[1]) /
                            this->stride[1];
    auto output_h = (inputs.dimension(1) + 2 * this->padding_h - kernel[0]) /
                            this->stride[0];
    auto num_patches = output_w * output_h;
    auto num_channels = inputs.dimension(4);


/*
    this->Z = this->X
            .extract_image_patches(this->kernel[0], this->kernel[1], this->stride[0],
                                   this->stride[1], this->dilation[0],
                                   this->dilation[1], this->padding)
            .reshape(Dims<3>(batches, num_patches,
                             filter_size * filter_size * num_channels))
            .contract(kernel.reshape(
                              Dims<2>(filters, filter_size * filter_size * channels)),
                      ContractDim{Axes(2, 1)})
            .reshape(Dims<4>(batch, output_w_h, output_w_h, filters));
*/

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