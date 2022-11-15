//
// Created by macross on 10/10/22.
//

#include "maxpooling2d.hpp"

namespace orion
{

MaxPooling2D::MaxPooling2D(const PoolSize &pool, const Stride &stride,
                           Padding padding) :
        pool(pool),
        stride(stride),
        padding(padding)
{
    // nothing to do
}


MaxPooling2D::MaxPooling2D(const PoolSize &pool, Padding padding)
        : pool(pool),
          padding(padding),
          stride(Stride(1, 1)),
          dilation(Dilation(1, 1))
{
    // nothing to do
}


void MaxPooling2D::Forward(const Tensor<4> &inputs)
{
    this->X = inputs;

    Eigen::Index batches = inputs.dimension(0);
    Eigen::Index channels = inputs.dimension(3);
    Eigen::Index input_h = inputs.dimension(1);
    Eigen::Index input_w = inputs.dimension(2);

    Eigen::Index pad_h = 0;
    Eigen::Index pad_w = 0;

    if (this->padding == Eigen::PADDING_SAME) {
        // compute total padding required
        pad_h = (input_h - 1) * this->stride.height() -
                input_h + this->dilation.height() * (this->pool.height() - 1) + 1;
        pad_w = (input_w - 1) * this->stride.width() -
                input_w + this->dilation.width() * (this->pool.width() - 1) + 1;
    }

    // note that in the formula here, pad_h already contains a "2"
    Eigen::Index output_h =
            ((input_h + pad_h - this->dilation.height() * (this->pool.height() - 1) -
              1) / this->stride.height()) + 1;
    Eigen::Index output_w =
            ((input_w + pad_w - this->dilation.width() * (this->pool.width() - 1) -
              1) / this->stride.width()) + 1;

    this->Z = inputs
            .extract_image_patches(
                    this->pool.height(), this->pool.width(),
                    this->stride.height(), this->stride.width(),
                    this->dilation.height(), this->dilation.width(),
                    1, 1,
                    pad_h / 2, pad_h - pad_h / 2,
                    pad_w / 2, pad_w - pad_w / 2, 0.0)
            .maximum(Dims<2>(2, 3))
            .reshape(Dims<4>(batches, output_h, output_w, channels));
}


void MaxPooling2D::Forward(const Layer &prev)
{
    this->Forward(prev.GetOutput4D());
}


void MaxPooling2D::Backward(const LossFunction &loss_function)
{
    this->Backward(loss_function.GetGradients4D() /
                   static_cast<Scalar>(this->Z.dimensions().TotalSize() /
                                       this->Z.dimension(0)));
}


void MaxPooling2D::Backward(const Layer &next)
{
    this->Backward(next.GetInputGradients4D());
}


void MaxPooling2D::Backward(const Tensor<4> &gradients)
{
    Eigen::Index batch = this->X.dimension(0);
    Eigen::Index img_h = this->X.dimension(1);
    Eigen::Index img_w = this->X.dimension(2);
    Eigen::Index channels = this->X.dimension(3);

    // gap values are the amount of indices to skip in the input tensor to reach the
    // next row as the pooling window slides left to right and stops to stay within bounds
    Eigen::Index gap_w = this->pool.width() - 1;
    Eigen::Index gap_h = this->pool.height() - 1;

    Eigen::Index pool_size = pool.TotalSize();

    Eigen::Index output_h = gradients.dimension(1);
    Eigen::Index output_w = gradients.dimension(2);


    // input_flatten the output gradients into [N*H*W,C]
    Tensor<2> grad_flatten = gradients.reshape(Dims<2>(
            gradients.dimension(0) * gradients.dimension(1) * gradients.dimension(2),
            gradients.dimension(3)));

    // input_flatten the layer input into [N*H*W,C]
    Tensor<2> input_flatten = this->X.reshape(Dims<2>(
            this->X.dimension(0) * this->X.dimension(1) * this->X.dimension(2),
            this->X.dimension(3)));

    // create a zeroed flattened version of the input gradients that will become dL / dX
    Tensor<2> dL_dX_flatten(input_flatten.dimensions());
    dL_dX_flatten.setZero();




    // let patch be the sections on which the pool slides over
    // iterate through each gradient value
    for (Eigen::Index patch = 0; patch < grad_flatten.dimension(0); patch++) {
        // find the index of the patch's first value in flattened input tensor
        Eigen::Index patch_start =
                patch + (patch / output_w) * gap_w +
                (patch / pool_size) * img_w * gap_h;

        Scalar max_val = INT_MIN;
        Eigen::Index max_index = -1;

        // iterate through the patch's values (per channel) and find the index
        // of the patch's max value as it exists in the flattened input tensor,
        // apply the current gradient value (indexed as patch) to that index in the
        // flattened dL_dX tensor
        for (Eigen::Index c = 0; c < channels; c++) {
            for (Eigen::Index pool_index = 0; pool_index < pool_size; pool_index++) {

                Eigen::Index actual_index =
                        patch_start + pool_index +
                        (pool_index / this->pool.width()) * gap_w;

                if (input_flatten(actual_index) > max_val) {
                    max_val = input_flatten(actual_index);
                    max_index = actual_index;
                }
            }

            dL_dX_flatten(max_index, c) += grad_flatten(patch, c);
        }

    }

    // dL / dX now contains all that gradient values routed back to their respective max indices
    // reshape it back to the input tensor's dimensions
    this->dL_dX = dL_dX_flatten.reshape(this->X.dimensions());
}


void MaxPooling2D::Update(Optimizer &)
{
    // max pooling has no parameter to update
}


const Tensor<4> &MaxPooling2D::GetOutput4D() const
{
    return this->Z;
}


const Tensor<4> &MaxPooling2D::GetInputGradients4D() const
{
    return this->dL_dX;
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