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
    this->name = "maxpooling2d";
}


MaxPooling2D::MaxPooling2D(const PoolSize &pool, Padding padding)
        : MaxPooling2D(pool, pool, padding)
{
    // calls the constructor with all arguments
}


void MaxPooling2D::Forward(const Tensor<4> &inputs)
{
    this->X = inputs;

    this->Z = MaxPooling2DForward(
            inputs, this->pool,
            this->stride, this->dilation, this->padding);
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
    this->dL_dZ = gradients;
}


void MaxPooling2D::Update(Optimizer &)
{
    // max pooling has no parameters to update
}


const Tensor<4> &MaxPooling2D::GetOutput4D() const
{
    return this->Z;
}


Tensor<4> MaxPooling2D::GetInputGradients4D() const
{
    return MaxPooling2D::MaxPooling2DBackwardInput(
            this->X, this->dL_dZ, this->pool,
            this->stride, this->dilation, this->padding);
}


int MaxPooling2D::GetInputRank() const
{
    return 4;
}


int MaxPooling2D::GetOutputRank() const
{
    return 4;
}


Tensor<4> MaxPooling2D::MaxPooling2DForward(
        const Tensor<4> &inputs, const PoolSize &pool,
        const Stride &stride, const Dilation &dilation, Padding padding)
{
    Eigen::Index batches = inputs.dimension(0);
    Eigen::Index channels = inputs.dimension(3);
    Eigen::Index input_h = inputs.dimension(1);
    Eigen::Index input_w = inputs.dimension(2);

    Eigen::Index pad_h = 0;
    Eigen::Index pad_w = 0;
    Eigen::Index output_h = 0;
    Eigen::Index output_w = 0;

    if (padding == Eigen::PADDING_SAME) {
        // compute total padding required
        output_h = std::ceil(static_cast<Scalar>(input_h) /
                             static_cast<Scalar>(stride.height()));
        output_w = std::ceil(static_cast<Scalar>(input_w) /
                             static_cast<Scalar>(stride.width()));

        pad_h = (output_h - 1) * stride.height() -
                input_h + dilation.height() * (pool.height() - 1) + 1;
        pad_w = (output_w - 1) * stride.width() -
                input_w + dilation.width() * (pool.width() - 1) + 1;
    }
    else {
        // note that in the formula here, pad_h & pad_w already contains a "2" factor
        output_h = 1 + (input_h + pad_h -
                        dilation.height() * (pool.height() - 1) - 1) /
                       stride.height();
        output_w = 1 + (input_w + pad_w -
                        dilation.width() * (pool.width() - 1) - 1) /
                       stride.width();
    }

    // 1. extract image patches to get [N,P,H,W,C], each patch is the same size as pool
    // 2. find the max value along the height and width dimensions to get [N,P,C]
    // 3. patch dimension P contains the max values, reshape back to [N,H,W,C]
    return inputs
            .extract_image_patches(
                    pool.height(), pool.width(),
                    stride.height(), stride.width(),
                    dilation.height(), dilation.width(),
                    1, 1,
                    pad_h / 2, pad_h - pad_h / 2,
                    pad_w / 2, pad_w - pad_w / 2,
                    std::numeric_limits<Scalar>::lowest())
            .maximum(Dims<2>(2, 3))
            .reshape(Dims<4>(batches, output_h, output_w, channels));
}


Tensor<4> MaxPooling2D::MaxPooling2DBackwardInput(
        const Tensor<4> &inputs, const Tensor<4> &gradients, const PoolSize &pool,
        const Stride &stride, const Dilation &dilation, Padding padding)
{
    Eigen::Index batches = inputs.dimension(0);
    Eigen::Index input_h = inputs.dimension(1);
    Eigen::Index input_w = inputs.dimension(2);
    Eigen::Index channels = inputs.dimension(3);

    Eigen::Index grad_h = gradients.dimension(1);
    Eigen::Index grad_w = gradients.dimension(2);

    Eigen::Index pad_h = 0;
    Eigen::Index pad_w = 0;

    Eigen::array<std::pair<int, int>, 4> pad_dims;

    if (padding == Eigen::PADDING_SAME) {
        // compute total padding used in forward propagation
        Eigen::Index output_h = std::ceil(static_cast<Scalar>(input_h) /
                                          static_cast<Scalar>(stride.height()));
        Eigen::Index output_w = std::ceil(static_cast<Scalar>(input_w) /
                                          static_cast<Scalar>(stride.width()));

        pad_h = (output_h - 1) * stride.height() -
                input_h + dilation.height() * (pool.height() - 1) + 1;
        pad_w = (output_w - 1) * stride.width() -
                input_w + dilation.width() * (pool.width() - 1) + 1;
    }

    // pad_dims the input tensor as was done during forward propagation
    pad_dims[0] = std::make_pair(0, 0); // don't pad_dims batch
    pad_dims[1] = std::make_pair(pad_w / 2, pad_w - pad_w / 2); // width padding
    pad_dims[2] = std::make_pair(pad_h / 2, pad_h - pad_h / 2); // height padding
    pad_dims[3] = std::make_pair(0, 0); // don't pad_dims channels

    Tensor<4> inputs_padded = inputs.pad(pad_dims,
                                         std::numeric_limits<Scalar>::lowest());

    // this will become dL / dX which is passed to the previous layer as dL / dZ
    Tensor<4> input_gradients(inputs_padded.dimensions());
    input_gradients.setZero();

    // loop through all values of the gradient tensor
    for (Eigen::Index n = 0; n < batches; n++) {
        for (Eigen::Index h = 0; h < grad_h; h++) {
            // set the pool window's first and last row & col indices
            Eigen::Index pool_start_r = h * stride.height();
            Eigen::Index pool_end_r = pool_start_r + pool.height();

            for (Eigen::Index w = 0; w < grad_w; w++) {
                Eigen::Index pool_start_c = w * stride.width();
                Eigen::Index pool_end_c = pool_start_c + pool.width();

                // in each channel, find the pool window's maximum value's row,col
                for (Eigen::Index c = 0; c < channels; c++) {
                    Eigen::Index max_val_row = pool_start_r;
                    Eigen::Index max_val_col = pool_start_c;
                    Scalar max_val = inputs_padded(n, pool_start_r, pool_start_c, c);

                    // starting at the first value, visit each value of the
                    // pool window from left to right, row by row
                    for (Eigen::Index pool_row = pool_start_r;
                         pool_row < pool_end_r; pool_row++) {
                        for (Eigen::Index pool_col = pool_start_c;
                             pool_col < pool_end_c; pool_col++) {
                            if (inputs_padded(n, pool_row, pool_col, c) > max_val) {
                                max_val = inputs_padded(n, pool_row, pool_col, c);
                                max_val_row = pool_row;
                                max_val_col = pool_col;
                            }
                        }
                    }

                    // the current gradient value gets added to the position of the
                    // input tensor's max value for the current pool window
                    input_gradients(n, max_val_row, max_val_col, c) +=
                            gradients(n, h, w, c);
                }
            }
        }
    }

    // return the input gradients without the padding
    return input_gradients.slice(
            Dims<4>(0, pad_h / 2, pad_w / 2, 0),
            Dims<4>(batches, input_h, input_w, channels));
}

} // namespace orion