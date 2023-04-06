//
// Created by macross on 9/1/22.
//

#include "embedding.hpp"

namespace fl
{

Embedding::Embedding(Eigen::Index vocab_size, Eigen::Index embedding_dim,
              Eigen::Index input_len,
                     const Initializer<2> &initializer)
        : embed_dims(embedding_dim),
          input_len(input_len)
{
    this->name = "embedding";
    this->dL_dw.setZero();
    this->dL_dX.resize(0, 0);
    Dims<2>(vocab_size, embedding_dim);
    this->w.resize(vocab_size, embedding_dim);
    this->w.device(this->device) = initializer.Initialize(
            Tensor<2>::Dimensions(vocab_size, embedding_dim),
            input_len, embedding_dim);
    this->dL_dw.resize(w.dimensions());
}


void Embedding::Forward(const Tensor<2> &inputs)
{
    fl_assert(inputs.dimension(1) == this->input_len,
              this->name << " Embedding::Forward expected " << this->input_len
                         << " input features, got " << inputs.dimension(0));

    // store the input so backpropagation knows which weight rows to update
    this->X.resize(inputs.dimensions());
    this->X.device(this->device) = inputs;

    // layer output shape is (batch=input.cols, row=input.rows, embed_dim)
    this->Z.resize(Dims<3>(
            inputs.dimension(0), inputs.dimension(1), this->embed_dims));

    Dims<3> z_extent(1, 1, this->embed_dims);

    for (Eigen::Index batch = 0; batch < inputs.dimension(0); batch++) {
        for (Eigen::Index i = 0; i < inputs.dimension(1); i++) {
            Dims<3> z_offset(batch, i, 0);

            this->Z.slice(z_offset, z_extent).device(this->device) =
                    this->w.chip(static_cast<Eigen::Index>(inputs(batch, i)), 0)
                            .reshape(z_extent);
        }
    }
}


const Tensor<2> &Embedding::GetInputGradients2D()
{
    return this->dL_dX; // return dummy gradients
}


void Embedding::Backward(Layer &next)
{
    this->Backward(next.GetInputGradients3D());
}


void Embedding::Backward(const Tensor<3> &gradients)
{
    fl_assert(this->Z.dimensions() == gradients.dimensions(),
              this->name << "::Backward expected gradient dimension "
                         << this->Z.dimensions() << ", instead got "
                         << gradients.dimensions());

    // each vector from the gradients' 3rd dimension corresponds to a
    // value from the input tensor which indicates the dL_dw row that vector
    // gets added to
    this->dL_dw.setZero();
    Dims<3> z_extent(1, 1, this->embed_dims);

    for (Eigen::Index batch = 0; batch < gradients.dimension(0); batch++) {
        for (Eigen::Index i = 0; i < gradients.dimension(1); i++) {
            Dims<3> z_offset(batch, i, 0);

            this->dL_dw.chip(static_cast<Eigen::Index>(this->X(batch, i)), 0)
                    .device(device) += gradients.slice(z_offset, z_extent)
                    .reshape(Dims<1>(this->embed_dims));
        }
    }
}


void Embedding::Update(Optimizer &optimizer)
{
    // with Embedding(10000, 64, 16), input(64, 16) x1000, SGD,
    // it was ~37.5% faster updating weights directly than using dL_dw
    optimizer.Minimize(this->w, this->dL_dw);
}


const Tensor<3> &Embedding::GetOutput3D() const
{
    return this->Z;
}


std::vector<fl::Tensor<2>> Embedding::GetWeights2D() const
{
    return {this->w};
}


std::vector<Tensor<2>> Embedding::GetWeightGradients2D() const
{
    return {this->dL_dw};
}


void Embedding::SetWeights(const std::vector<Tensor<2>> &weights)
{
    if (weights.front().dimensions() != this->w.dimensions()) {
        std::ostringstream error_msg;
        error_msg << this->name << " expected weights dimensions " <<
                  this->w.dimensions() << ", got " << weights.front().dimensions()
                  << "\n";
        throw std::invalid_argument(error_msg.str());
    }

    this->w.device(this->device) = weights.front();
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