//
// Created by macross on 10/10/22.
//

#include "maxpooling2d.hpp"

namespace orion
{

MaxPooling2D::MaxPooling2D(const PoolSize &pool_dim, const Stride &stride_dim) :
        pool_dim(pool_dim),
        stride_dim(stride_dim)
{
    // assumes PADDING_VALID
}


void MaxPooling2D::Forward(const Tensor<4> &inputs)
{
    auto batches = inputs.dimension(0);
    auto channels = inputs.dimension(3);

    auto output_h =
            ((inputs.dimension(1) - this->pool_dim[0]) / this->stride_dim[0]) + 1;
    auto output_w =
            ((inputs.dimension(2)  - this->pool_dim[1]) / this->stride_dim[1]) + 1;

    this->Z = inputs
            .extract_image_patches(
                    this->pool_dim[0], this->pool_dim[1],
                    this->stride_dim[0], this->stride_dim[1],
                    1, 1, Eigen::PADDING_VALID)
            .maximum(Dims<2>(2, 3))
            .reshape(Dims<4>(batches, output_h, output_w, channels));
}


void MaxPooling2D::Forward(const Layer &prev)
{
    Layer::Forward(prev);
}


void MaxPooling2D::Backward(const LossFunction &loss_function)
{
    Layer::Backward(loss_function);
}


void MaxPooling2D::Backward(const Layer &next)
{
    Layer::Backward(next);
}


const Tensor<4> &MaxPooling2D::GetOutput4D() const
{
    return this->Z;
}


const Tensor<4> &MaxPooling2D::GetInputGradients4D() const
{
    return Layer::GetInputGradients4D();
}


int MaxPooling2D::GetInputRank() const
{
    return 4;
}


int MaxPooling2D::GetOutputRank() const
{
    return 4;
}
} // namespace orion