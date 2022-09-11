//
// Created by macross on 8/7/22.
//

#include "dense.hpp"

namespace orion
{

template<typename Activation>
Dense<Activation>::Dense(int inputs, int outputs, bool use_bias,
                         const Initializer &initializer) :
        use_bias(use_bias)
{
    this->name = "dense";
    this->w = initializer.Initialize(outputs, inputs);

    if (this->use_bias) {
        this->b.resize(outputs, 1);
        this->b.setZero();
    }
}


template<typename Activation>
void Dense<Activation>::Forward(const Tensor2D &input)
{
    orion_assert(this->w.dimension(1) == input.dimension(0), this->name <<
            " Dense::Forward EXPECTED " << this->w.dimension(1)
            << " INPUT FEATURES, INSTEAD GOT "
            << input.dimension(0));

    // Z = W * X + b
    this->X = input;
    this->Z = this->w.contract(this->X, this->matmul);

    this->A = Activation::Activate(this->Z);
    //std::cout << this->name << " A=\n" << this->A << "\n";
}


template<typename Activation>
void Dense<Activation>::Forward(const Layer &prev)
{
    this->Forward(prev.GetOutput2D());
}


template<typename Activation>
Tensor2D Dense<Activation>::operator()(const Tensor2D &tensor) const
{
    orion_assert(this->w.dimension(1) == tensor.dimension(0),
                 "Dense::Forward EXPECTED " << this->w.dimension(1)
                         << " INPUT FEATURES, INSTEAD GOT "
                         << tensor.dimension(0));

    return Activation::Activate(this->w.contract(tensor, this->matmul));
}


template<typename Activation>
void Dense<Activation>::Backward(const Layer &next) // hidden layer backward
{
    std::cout << "next weights:\n" << next.GetWeights() << "\n";
    std::cout << "next dL/dZ:\n" << next.GetInputGradients2D() << "\n";
    std::cout << "sigmoid derivative:\n" << Activation::Activate(this->Z) << "\n";
    this->Backward(next.GetWeights().contract(next.GetInputGradients2D(),
                                              ContractDim{Axes{0, 0}}));
    //std::terminate();
}


template<typename Activation>
void Dense<Activation>::Backward(const Loss &loss) // output backward
{
    //std::cout << "loss gradients:\n" << loss.GetGradients2D() << "\n";
    this->Backward(loss.GetGradients2D());
}


template<typename Activation>
void Dense<Activation>::Backward(const Tensor2D &gradients)
{
    orion_assert(gradients.dimensions() == this->Z.dimensions(),
                 this->name << " Dense::Backward expected "
                         << this->Z.dimensions() << ", got "
                         << gradients.dimensions() << " instead");

    // dL/dw = dL/dz * dz/dw * 1/m = (dL/da * da/dz) * dz/dw * 1/m,
    this->dL_dZ = gradients * Activation::Gradients(this->Z);
    this->dL_dw = this->dL_dZ.contract(this->X, ContractDim{Axes(1, 1)}) /
                  (Scalar) this->w.dimension(0);


    orion_assert(this->w.dimensions() == this->dL_dw.dimensions(), this->name <<
            " Dense::Backward weights dimensions " << this->w.dimensions()
            << " do not match weights gradient dimensions "
            << this->dL_dw.dimensions());

    if (this->use_bias) {
        // dL/db = dL/dZ, if batch size > 1 then sum along the 1st dimension (col)
        this->dL_db = this->dL_dZ.sum(Eigen::array<int, 1>{1}) /
                      (Scalar) this->w.dimension(0);

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
const Tensor2D &Dense<Activation>::GetOutput2D() const
{
    return this->A;
}


template<typename Activation>
const Tensor<2> &Dense<Activation>::GetInputGradients2D() const
{
    return this->dL_dZ;
}


template<typename Activation>
Tensor2D Dense<Activation>::GetGradients() const
{
    return this->w.contract(this->dL_dZ, ContractDim{Axes(0, 0)});
}


template<typename Activation>
const Tensor2D &Dense<Activation>::GetWeights() const
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
        error_msg << this->name << " expected weights dimensions " <<
                this->w.dimensions() << ", got " << weights.dimensions()
                << "\n";
        throw std::invalid_argument(error_msg.str());
    }

    this->w = weights;
}


template<typename Activation>
Tensor2D &Dense<Activation>::Bias()
{
    return this->b;
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