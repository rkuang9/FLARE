//
// Created by macross on 9/1/22.
//

#include "embedding.hpp"

namespace orion
{

Embedding::Embedding(int vocab_size, int embedding_dim, int input_length,
                     const Initializer &initializer)
        : w(initializer.Initialize(vocab_size, embedding_dim)),
          embed_dims(embedding_dim),
          input_len(input_length)
{
    this->name = "embedding";
    this->dL_dw.resize(w.dimensions());
    this->dL_dw.setZero();
}


void Embedding::Forward(const Tensor<2> &input)
{
    orion_assert(input.dimension(0) == this->input_len,
                 this->name << " Embedding::Forward expected " << this->input_len
                         << " input features, got " << input.dimension(0));

    // store the input so backpropagation knows which weight rows to update
    this->X = input;

    // layer output shape is (batch=input.cols, row=input.rows, embed_dum)
    this->Z.resize(Tensor<3>::Dimensions(
            input.dimension(1), input.dimension(0), this->embed_dims));

    // from the input tensor values construct output tensor Z
    for (Eigen::Index col = 0; col < input.dimension(1); col++) {
        // col also denotes the batch dimension of the output tensor Z
        for (Eigen::Index row = 0; row < input.dimension(0); row++) {
            // target a weight-row slice to be placed into output Z
            Eigen::array<Eigen::Index, 2> w_offset{Eigen::Index(input(row, col)), 0};
            Eigen::array<Eigen::Index, 2> w_extent{1, this->embed_dims};

            // identify the output slice to set with the weight slice with
            Eigen::array<Eigen::Index, 2> z_offset{row, 0};
            Eigen::array<Eigen::Index, 2> z_extent{1, this->embed_dims};

            this->Z.chip(col, 0).slice(z_offset, z_extent) =
                    this->w.slice(w_offset, w_extent);
        }
    }
}


void Embedding::Backward(const Layer &next)
{
    // divide by the batch size
    this->dL_dZ = next.GetInputGradients2D() / (Scalar) this->Z.dimension(0);
    std::cout << "received gradients\n" << next.GetInputGradients2D() << "\n";
    this->Backward();
}


// TODO: not tested yet
void Embedding::Backward(const Loss &loss_function)
{
    this->dL_dZ = loss_function.GetGradients2D() / (Scalar) this->X.dimension(0);
    this->Backward();
}


void Embedding::Backward()
{
    /*orion_assert(this->dL_dZ.dimensions() == this->X.dimensions(),
                 this->name + " Embedding::Backward expected gradient dimension "
                         << this->X.dimensions() << ", received "
                         << this->dL_dZ.dimensions());*/

    this->dL_dw.setZero();

    for (Eigen::Index col = 0; col < this->X.dimension(1); col++) {
        for (Eigen::Index row = 0; row < this->X.dimension(0); row++) {
            this->dL_dw.chip(Eigen::Index(this->X(row, col)), 0) =
                    this->dL_dw.chip(Eigen::Index(this->X(row, col)), 0) +
                    this->dL_dZ(row, col);
        }
    }
}


void Embedding::Update(Optimizer &optimizer)
{
    optimizer.Minimize(this->w, this->dL_dw);
}


const Tensor<2> &Embedding::GetInputGradients2D() const
{
    return this->dL_dZ;
}


const Tensor<3> &Embedding::GetOutput3D() const
{
    return this->Z;
}


const Tensor<2> &Embedding::GetWeights() const
{
    return this->w;
}


const Tensor<2> &Embedding::GetWeightGradients() const
{
    return this->dL_dw;
}


void Embedding::SetWeights(const Tensor<2> &weights)
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


int Embedding::GetInputRank() const
{
    return 2; // column vector of inputs stacked sideways as a matrix
}


int Embedding::GetOutputRank() const
{
    return 3; // matrices of word embeddings stacked in the batch dimension
}


Tensor2D Embedding::operator()(const Tensor2D &tensor) const
{

}

} // namespace orion