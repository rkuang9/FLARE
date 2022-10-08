//
// Created by macross on 8/7/22.
//

#include "dense.hpp"

namespace orion
{

template<typename Activation>
Dense<Activation>::Dense(int inputs, int outputs, bool use_bias,
                         const Initializer<2> &initializer) :
        use_bias(use_bias),
        w(initializer.Initialize(
                Tensor<2>::Dimensions(outputs, inputs), inputs, outputs))
{
    this->name = "dense";

    if (this->use_bias) {
        throw std::invalid_argument("bias not available");
        this->b.resize(outputs, 1);
        this->b.setZero();
    }
}


template<typename Activation>
void Dense<Activation>::Forward(const Tensor<2> &input)
{
    orion_assert(this->w.dimension(1) == input.dimension(0), this->name <<
            " Dense::Forward EXPECTED " << this->w.dimension(1)
            << " INPUT FEATURES, INSTEAD GOT "
            << input.dimension(0));

    // Z = w * X + b
    this->X = input;
    this->Z = this->w.contract(this->X, ContractDim{Axes(1, 0)});

    if (this->use_bias) {
        // broadcast bias into the shape of Z
        // TODO: broadcast accepts Tensor<2>::Dimensions()
        this->Z += this->b.broadcast(
                Eigen::array<Eigen::Index, 2>({1, input.dimension(1)}));
    }

    this->A = Activation::Activate(this->Z);
}


template<typename Activation>
void Dense<Activation>::Forward(const Layer &prev)
{
    this->Forward(prev.GetOutput2D());
}


template<typename Activation>
Tensor<2> Dense<Activation>::operator()(const Tensor<2> &tensor) const
{
    orion_assert(this->w.dimension(1) == tensor.dimension(0),
                 "Dense::Forward EXPECTED " << this->w.dimension(1)
                         << " INPUT FEATURES, GOT " << tensor.dimension(0));
    Tensor<2> z = this->w.contract(tensor, ContractDim{Axes(1, 0)});
    return Activation::Activate(z);
}


template<typename Activation>
void Dense<Activation>::Backward(const Layer &next) // hidden layer backward
{
    // (next.weights.transpose * next.dL_dZ) hadamard g'(this->Z)
    this->dL_dZ = next.GetWeights().contract(
            next.GetInputGradients2D(), ContractDim{Axes{0, 0}}) *
                  Activation::Gradients(this->Z);
    this->Backward();
}


template<typename Activation>
void Dense<Activation>::Backward(const LossFunction &loss) // output backward
{
    // divide by num output units
    this->dL_dZ = loss.GetGradients2D() * Activation::Gradients(this->Z) /
                  (Scalar) this->w.dimension(0);
    this->Backward();
}


template<typename Activation>
void Dense<Activation>::Backward()
{
    // dL / dw = (dL / dZ) * (dZ / dw) * (1 / m) where m = batch size
    this->dL_dw = this->dL_dZ.contract(this->X, ContractDim{Axes(1, 1)})
                  / (Scalar) this->X.dimension(1); // divide by batch size


    orion_assert(this->w.dimensions() == this->dL_dw.dimensions(), this->name <<
            " Dense::Backward weights dimensions " << this->w.dimensions()
            << " do not match weights gradient dimensions "
            << this->dL_dw.dimensions());

    if (this->use_bias) {
        throw std::invalid_argument("bias not available yet");
        // dL/db = dL/dZ, if batch size > 1 then sum along the 1st dimension (col)
        this->dL_db = this->dL_dZ.sum(Eigen::array<int, 1>{1}).reshape(
                Tensor<2>::Dimensions(this->dL_dZ.dimension(0), 1)) /
                      (Scalar) this->dL_dZ.dimension(1);

        orion_assert(this->b.dimensions() == this->dL_db.dimensions(),
                     "Dense::Backward BIAS DIMENSIONS " << this->b.dimensions()
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
const Tensor<2> &Dense<Activation>::GetInputGradients2D() const
{
    return this->dL_dZ;
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

} // namespace orion