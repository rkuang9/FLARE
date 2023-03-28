//
// Created by macross on 9/1/22.
//

#include "embedding.hpp"

namespace fl
{

Embedding::Embedding(int vocab_size, int embedding_dim, int input_length,
                     const Initializer<2> &initializer)
        : embed_dims(embedding_dim),
          input_len(input_length)
{
    this->name = "embedding";
    this->dL_dw.setZero();

    this->w.resize(vocab_size, embedding_dim);
    this->w.device(this->device) = initializer.Initialize(
            Tensor<2>::Dimensions(vocab_size, embedding_dim),
            input_length, embedding_dim);
    this->dL_dw.resize(w.dimensions());
}


void Embedding::Forward(const Tensor<2> &input)
{
    fl_assert(input.dimension(1) == this->input_len,
              this->name << " Embedding::Forward expected " << this->input_len
                         << " input features, got " << input.dimension(0));

    // store the input so backpropagation knows which weight rows to update
    this->X.resize(input.dimensions());
    this->X.device(this->device) = input;

    // layer output shape is (batch=input.cols, row=input.rows, embed_dim)
    this->Z.resize(Tensor<3>::Dimensions(
            input.dimension(1), input.dimension(0), this->embed_dims));

    // TODO: see if chip() can make this code less verbose
    // from the input tensor values construct output tensor Z
    for (Eigen::Index col = 0; col < input.dimension(1); col++) {
        // col also denotes the batch dimension of the output tensor Z
        for (Eigen::Index row = 0; row < input.dimension(0); row++) {
            // target a weight-row slice to be placed into output Z
            Eigen::array<Eigen::Index, 2> w_offset {Eigen::Index(input(row, col)),
                                                    0};
            Eigen::array<Eigen::Index, 2> w_extent {1, this->embed_dims};

            // identify the output slice to set with the weight slice with
            Eigen::array<Eigen::Index, 2> z_offset {row, 0};
            Eigen::array<Eigen::Index, 2> z_extent {1, this->embed_dims};

            this->Z.chip(col, 0).slice(z_offset, z_extent) =
                    this->w.slice(w_offset, w_extent);
        }
    }
}


void Embedding::Backward(Layer &next)
{
    this->Backward(next.GetInputGradients3D());
}


// TODO: not tested yet
void Embedding::Backward(const Tensor<3> &gradients)
{
    throw std::logic_error("Embedding as output layer not supported");
    // TODO: implement loss_function.GetGradients3D()
    // TODO: division is moved to loss function
    this->dL_dZ.device(this->device) = gradients;

    fl_assert(this->dL_dZ.dimensions() == this->Z.dimensions(),
              this->name << " Embedding::Backward expected gradient dimensions "
                         << this->Z.dimensions() <<
                         ", instead got " << this->dL_dZ.dimensions());

    this->dL_dw.setZero();

    // dimension(0), dimension(1) of dL_dZ denote the batch size, input length
    // dimension(0) is also tied to the input tensor X's dimension(1) (aka batch size)
    for (Eigen::Index batch = 0; batch < this->dL_dZ.dimension(0); batch++) {
        for (Eigen::Index row = 0; row < this->dL_dZ.dimension(1); row++) {
            // each row of dL_dZ is added to dL_dw w.r.t. the input tensor's
            // and divided by batch size
            this->dL_dw.chip((Eigen::Index) this->X(row, batch), 0) +=
                    this->dL_dZ.chip(batch, 0).chip(row, 0) /
                    (Scalar) this->X.dimension(0);
        }
    }
}


void Embedding::Backward()
{

}


void Embedding::Update(Optimizer &optimizer)
{
    optimizer.Minimize(this->w, this->dL_dw);
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

    this->w.device(this->device) = weights;
}


int Embedding::GetInputRank() const
{
    return 2; // [batch, words]
}


int Embedding::GetOutputRank() const
{
    return 3; // [batch, word, embedding]
}


void Embedding::Save(const std::string &path)
{
    std::ofstream output_file(path);
    output_file.precision(15);

    if (!output_file.is_open()) {
        throw std::invalid_argument(
                this->name + "::Save INVALID FILE PATH: " + path);
    }

    // flatten the weights and write it to the file with a white space delimiter
    Tensor<1> flatten = this->w.reshape(Dims<1>(this->w.size()));

    std::vector<Scalar> as_vector(flatten.data(), flatten.data() + flatten.size());
    std::copy(as_vector.begin(), as_vector.end(),
              std::ostream_iterator<Scalar>(output_file, " "));
    output_file.close();
}


void Embedding::Load(const std::string &path)
{
    std::ifstream read_weights(path);

    if (!read_weights.is_open()) {
        throw std::invalid_argument(this->name + "::Load " + path + " NOT FOUND");
    }

    std::vector<Scalar> as_vector;
    std::copy(std::istream_iterator<Scalar>(read_weights),
              std::istream_iterator<Scalar>(), std::back_inserter(as_vector));
    read_weights.close();

    if (as_vector.size() != this->w.size()) {
        std::ostringstream error_msg;
        error_msg << this->name << "::Load " << path << " EXPECTED "
                  << this->w.dimensions() << "=" << this->w.size() << " VALUES, GOT "
                  << as_vector.size() << " INSTEAD";
        throw std::invalid_argument(error_msg.str());
    }

    // reshape the flattened tensor back to expected weights dimensions
    this->w = TensorMap<2>(as_vector.data(), this->w.dimensions());
}

} // namespace fl