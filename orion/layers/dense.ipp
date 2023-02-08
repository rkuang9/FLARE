//
// Created by macross on 8/7/22.
//

#include <type_traits>
#include "dense.hpp"

namespace orion
{

template<typename Activation>
Dense<Activation>::Dense(int inputs, int outputs, bool use_bias,
                         const Initializer<2> &initializer) :
        use_bias(use_bias),
        w(initializer.Initialize(Dims<2>(inputs, outputs), inputs, outputs))
{
    this->name = "dense";
    this->dL_dw.resize(this->w.dimensions());

    if (this->use_bias) {
        throw std::invalid_argument("bias not available");
        this->b.resize(outputs, 1);
        this->b.setZero();
    }
}


template<typename Activation>
void Dense<Activation>::Forward(const Tensor<2> &input)
{
    orion_assert(this->w.dimension(0) == input.dimension(1),
                 this->name << " Dense::Forward EXPECTED " << this->w.dimension(0)
                            << " INPUT FEATURES, INSTEAD GOT "
                            << input.dimension(1));

    // Z = Xw + b (same as Z = wX + b but with batch dims first in X)
    this->X = input;

    // resize output tensor to [batch, output_units]
    this->Z.resize(input.dimension(0), this->w.dimension(1));
    this->Z.template device(this->device) =
            this->X.contract(this->w, ContractDim {Axes(1, 0)});

    if (this->use_bias) {
        // broadcast bias into the shape of Z
        //this->Z += this->b.broadcast(
        //Eigen::array<Eigen::Index, 2>({1, input.dimension(1)}));
    }

    this->A = Activation::Activate(this->Z);

}


template<typename Activation>
void Dense<Activation>::Forward(const Layer &prev)
{
    this->Forward(prev.GetOutput2D());
}


template<typename Activation>
void Dense<Activation>::Backward(Layer &next) // hidden layer backward
{
    this->Backward(next.GetInputGradients2D());
}


template<typename Activation>
void Dense<Activation>::Backward(const Tensor<2> &gradients) // output backward
{
    if constexpr (std::is_same_v<Activation, Softmax>) {
        // softmax requires different calculations
        this->BackwardSoftmax(gradients);
    }
    else {
        this->dL_dZ.resize(this->Z.dimensions());
        this->dL_dZ.template device(this->device) =
                gradients * Activation::Gradients(this->Z);
    }

    // dL / dw = (dL / dZ) * (dZ / dw)
    this->dL_dw.template device(this->device) =
            this->X.contract(this->dL_dZ, ContractDim {Axes(0, 0)});


    orion_assert(this->w.dimensions() == this->dL_dw.dimensions(),
                 this->name << " Dense::Backward weights dimensions "
                            << this->w.dimensions()
                            << " do not match weights gradient dimensions "
                            << this->dL_dw.dimensions());

    if (this->use_bias) {
        throw std::invalid_argument("bias not available yet");
        // dL/db = dL/dZ, if batch size > 1 then sum along the 1st dimension (col)
        this->dL_db = this->dL_dZ.sum(Eigen::array<int, 1> {1}).reshape(
                Tensor<2>::Dimensions(this->dL_dZ.dimension(0), 1)) /
                      (Scalar) this->dL_dZ.dimension(1);

        orion_assert(this->b.dimensions() == this->dL_db.dimensions(),
                     this->name << " Dense::Backward BIAS DIMENSIONS "
                                << this->b.dimensions()
                                << " DO NOT MATCH BIAS GRADIENT DIMENSIONS "
                                << this->dL_db.dimensions());
    }
}


template<typename Activation>
void Dense<Activation>::Update(Optimizer &optimizer)
{
    optimizer.Minimize(this->w, this->dL_dw);

    if (this->use_bias) {
        optimizer.Minimize(this->b, this->dL_db);
    }
}


template<typename Activation>
const Tensor<2> &Dense<Activation>::GetOutput2D() const
{
    return this->A;
}


template<typename Activation>
const Tensor<2> &Dense<Activation>::GetInputGradients2D()
{
    this->dL_dX.resize(this->X.dimensions());
    this->dL_dX.template device(this->device) = this->dL_dZ.contract(
            this->w, ContractDim {Axes(1, 1)});
    return this->dL_dX;
}


template<typename Activation>
const Tensor<2> &Dense<Activation>::GetWeights() const
{
    return this->w;
}


template<typename Activation>
const Tensor<2> &Dense<Activation>::GetWeightGradients() const
{
    return this->dL_dw;
}


template<typename Activation>
void Dense<Activation>::SetWeights(const Tensor<2> &weights)
{
    if (weights.dimensions() != this->w.dimensions()) {
        std::ostringstream error_msg;
        error_msg << this->name << " Dense::SetWeights EXPECTED DIMENSIONS " <<
                  this->w.dimensions() << ", GOT " << weights.dimensions();
        throw std::invalid_argument(error_msg.str());
    }

    this->w = weights;
}


template<typename Activation>
const Tensor<2> &Dense<Activation>::GetBias() const
{
    return this->b;
}


template<typename Activation>
void Dense<Activation>::SetBias(const Tensor<2> &bias)
{
    if (bias.dimensions() != this->b.dimensions()) {
        std::ostringstream error_msg;
        error_msg << this->name << " Dense::SetBias EXPECTED DIMENSIONS " <<
                  this->b.dimensions() << ", GOT " << bias.dimensions();
        throw std::invalid_argument(error_msg.str());
    }

    this->b = bias;
}


template<typename Activation>
int Dense<Activation>::GetInputRank() const
{
    return 2;
}


template<typename Activation>
int Dense<Activation>::GetOutputRank() const
{
    return 2;
}


template<typename Activation>
void Dense<Activation>::BackwardSoftmax(const Tensor<2> &gradients)
{
    // TODO: performance improvement by eliminating for loop
    this->dL_dZ.resize(this->Z.dimensions());

    // pass in the layer activations to avoid recalculating the softmax values
    Tensor<3> softmax_grad = Softmax::Gradients(this->A);

    for (int batch = 0; batch < gradients.dimension(0); batch++) {
        // TODO: since it is using a threading device, does it need resizing?
        this->dL_dZ.chip(batch, 0).template device(this->device) =
                gradients.chip(batch, 0)
                        .contract(softmax_grad.chip(batch, 0),
                                  ContractDim {Axes(0, 1)});
    }
}

} // namespace orion