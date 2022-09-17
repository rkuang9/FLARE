//
// Created by macross on 9/1/22.
//

#include "embedding.hpp"

namespace orion
{

Embedding::Embedding(int vocab_size, int embedding_dim, int input_len,
                     const Initializer &initializer)
        : w(initializer.Initialize(vocab_size, embedding_dim)),
          embed_dim(embedding_dim)
{
    this->name = "embedding";
}


void Embedding::Forward(const Tensor<2> &input)
{
    // layer output shape is (row=input.rows, embed_dum, batch=input.cols)
    this->Z.resize(Tensor<3>::Dimensions(
            input.dimension(0), this->embed_dim, input.dimension(1)));

    // access input by column and extract a slice from vocabulary
    // as indicated by the column vector values
    for (Eigen::Index col = 0; col < input.dimension(1); col++) {
        for (Eigen::Index row = 0; row < input.dimension(0); row++) {
            Eigen::array<Scalar, 2> vocab_offset{input(row, col), 0};
            Eigen::array<Eigen::Index, 2> vocab_extent{1, w.dimension(1)};

            Eigen::array<Eigen::Index, 3> output_offset{row, 0, col};
            Eigen::array<Eigen::Index, 3> output_extent{1, this->w.dimension(1),
                                                        1};

            std::cout << "slice:\n" << this->w.slice(
                    vocab_offset, vocab_extent
            ) << "\n";

            this->Z.slice(output_offset, output_extent) = this->w.slice(
                    vocab_offset, vocab_extent
            ).reshape(Tensor3D::Dimensions(1, this->w.dimension(1), 1));
        }
    }
}


void Embedding::Backward(const Layer &next)
{

}


void Embedding::Update(Optimizer &optimizer)
{

}


const Tensor<3> &Embedding::GetOutput3D() const
{
    return this->Z;
}


const Tensor<2> &Embedding::GetWeights() const
{
    return this->w;
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