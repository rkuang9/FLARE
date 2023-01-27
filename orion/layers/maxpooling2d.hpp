//
// Created by macross on 10/10/22.
//

#ifndef ORION_MAXPOOLING2D_HPP
#define ORION_MAXPOOLING2D_HPP

#include "layer.hpp"

namespace orion
{

class MaxPooling2D : public Layer
{
public:
    /**
     * 2D max pooling layer, performs max pooling on images
     * @param pool      pooling dimensions, PoolSize(height, width)
     * @param stride    stride dimensions, Stride(height, width), defaults to pool
     * @param padding   padding type, Padding::PADDING_VALID or Padding::PADDING_SAME
     */
    MaxPooling2D(const PoolSize &pool, const Stride &stride, Padding padding);

    explicit MaxPooling2D(const PoolSize &pool,
                          Padding padding = Padding::PADDING_VALID);

    ~MaxPooling2D() override = default;


    /**
     * Max pooling forward operation, creates a new tensor by sliding, according to
     * the provided strides the, pool window across the input tensor, padded if necessary,
     * and extracting the maximum value
     *
     * @param inputs     input tensor in format NHWC
     * @param pool       PoolSize dimensions (h, w)
     * @param stride     Stride dimensions (h, w)
     * @param dilation   Dilation dimensions (h, w) (currently not supported)
     * @param padding    Padding enum, PADDING_VALID or PADDING_SAME
     * @return           tensor in format NHWC containing extracted max values
     */
    static auto MaxPooling2DForward(
            const Tensor<4> &inputs, const PoolSize &pool,
            const Stride &stride, const Dilation &dilation, Padding padding);


    /**
     * Forward propagation, extracts largest value within the
     * provided pool window
     *
     * While training, all inputs should have the same dimensions
     * While predicting, inputs may be larger than training inputs
     *
     * @param inputs   a rank 4 tensor
     */
    void Forward(const Tensor<4> &inputs) override;


    /**
     * Forward propagation for hidden layers, passes the previous layer's
     * output as input for Forward(inputs)
     *
     * @param prev   a reference to the previous layer
     */
    void Forward(const Layer &prev) override;

    /**
     * Backward propagation for output layers, compute the input gradients
     * by routing the output gradients back the indices that contributed to
     * the forward pass
     *
     * @param loss_function   a reference to the loss object
     */
    void Backward(const Tensor<4> &gradients) override;

    /**
     * Backward propagation for hidden layers, does the same as Backward(loss_function)
     * using gradients passed from the next layer
     *
     * @param next   a reference to the next layer
     */
    void Backward(const Layer &next) override;


    /**
     * Nothing to do, max pooling layers have no learnable parameters
     */
    void Update(Optimizer &) override;


    /**
     * @return   layer's activation values
     */
    const Tensor<4> &GetOutput4D() const override;


    /**
     * @return   layer's loss gradients w.r.t. pre-activated output (dL / dZ))
     */
    Tensor<4> GetInputGradients4D() const override;


    /**
     * @return   expected rank of forward propagation's input tensor
     */
    int GetInputRank() const override;


    /**
     * @return   expected rank of forward propagation's output tensor
     */
    int GetOutputRank() const override;

    /**
     * Max pooling backward operation, creates a tensor where gradient values
     * are routed to the indices of the input tensor's max values as found during the
     * forward pass
     *
     * @param inputs       input tensor in format NHWC
     * @param gradients    layer gradients passed from the next layer
     * @param pool         PoolSize dimensions (h, w)
     * @param stride       Stride dimensions (h, w)
     * @param dilation     Dilation dimensions (h, w)
     * @param padding      padding used in forward pass, PADDING_VALID or PADDING_SAME
     * @return             tensor in format NHWC with same dimensions as inputs
     */
    static Tensor<4> MaxPooling2DBackwardInput(
            const Tensor<4> &inputs, const Tensor<4> &gradients,
            const PoolSize &pool,
            const Stride &stride, const Dilation &dilation, Padding padding);


    static Dims<4> ForwardOutputDims(
            const Dims<4> &input_dims, const PoolSize &pool_size,
            const Stride &stride, const Dilation &dilation, Padding padding);

private:
    Tensor<4> X;
    Tensor<4> Z;
    Tensor<4> dL_dX;
    Tensor<4> dL_dZ;

    PoolSize pool;
    Stride stride;
    Dilation dilation = Dilation(1, 1);
    Padding padding;

    // multithreading
    Eigen::ThreadPool thread_pool;
    Eigen::ThreadPoolDevice device;

};

} // orion

#endif //ORION_MAXPOOLING2D_HPP
