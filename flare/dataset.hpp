//
// Created by Macross on 11/21/22.
//

#ifndef FLARE_DATASET_H
#define FLARE_DATASET_H

#include <vector>
#include <sstream>
#include <fstream>

#ifdef USING_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/eigen.hpp>
#endif

#include "flare/fl_types.hpp"

namespace fl
{

template<int SampleRank, int LabelRank>
class Dataset
{
public:
    Dataset(const Dims<SampleRank> &sample_dims, const Dims<LabelRank> &label_dims)
            : sample_dims(sample_dims),
              label_dims(label_dims)
    {
    }


    void Batch(int batch_size, bool shuffle = true, bool normalize = false)
    {
        if (shuffle) {
            this->ShufflePreBatch();
        }
        else {
            std::reverse(this->prebatch_samples.begin(),
                         this->prebatch_samples.end());
            std::reverse(this->prebatch_labels.begin(), this->prebatch_labels.end());
        }

        if (normalize) {
            this->NormalizePreBatch();
        }

        // create the batch dimensions by adding an extra dim to the front
        Dims<SampleRank + 1> batched_sample_dims;
        Dims<LabelRank + 1> batched_label_dims;
        batched_sample_dims[0] = batch_size;
        batched_label_dims[0] = batch_size;

        for (int i = 0; i < this->sample_dims.count; i++) {
            batched_sample_dims[i + 1] = this->sample_dims[i];
        }

        for (int j = 0; j < this->label_dims.count; j++) {
            batched_label_dims[j + 1] = this->label_dims[j];
        }

        int batches = this->prebatch_samples.size() / batch_size;
        int batches_remainder = this->prebatch_samples.size() % batch_size;

        if (batches_remainder > 0) {
            // create the last batch tensor where there's not enough for a full batch
            Dims<SampleRank + 1> remainder_batch_sample_dims = batched_sample_dims;
            Dims<LabelRank + 1> remainder_batch_label_dims = batched_label_dims;
            remainder_batch_sample_dims[0] = batches_remainder;
            remainder_batch_label_dims[0] = batches_remainder;

            Tensor<SampleRank + 1> remainder_sample(remainder_batch_sample_dims);
            Tensor<LabelRank + 1> remainder_label(remainder_batch_label_dims);

            for (int i = 0; i < batches_remainder; i++) {
                remainder_sample.chip(i, 0) = this->prebatch_samples.back();
                remainder_label.chip(i, 0) = this->prebatch_labels.back();
                this->prebatch_samples.pop_back();
                this->prebatch_labels.pop_back();
            }

            this->training_samples.push_back(std::move(remainder_sample));
            this->training_labels.push_back(std::move(remainder_label));
        }

        // create the batched tensors
        for (int batch = 0; batch < batches; batch++) {
            Tensor<SampleRank + 1> batched_sample(batched_sample_dims);
            Tensor<LabelRank + 1> batched_label(batched_label_dims);

            for (int batch_index = 0; batch_index < batch_size; batch_index++) {
                batched_sample.chip(batch_index, 0) = this->prebatch_samples.back();
                batched_label.chip(batch_index, 0) = this->prebatch_labels.back();
                this->prebatch_samples.pop_back();
                this->prebatch_labels.pop_back();
            }

            this->training_samples.push_back(std::move(batched_sample));
            this->training_labels.push_back(std::move(batched_label));
        }
    }


    void Add(const Tensor<SampleRank> &sample, const Tensor<LabelRank> &label)
    {
        // check that the sample and labels have the expected dimensions
        if (sample.dimensions() != this->sample_dims ||
            label.dimensions() != this->label_dims) {
            std::ostringstream error_msg;
            error_msg << "Dataset::Add EXPECTED SAMPLE,LABEL DIMENSIONS "
                      << this->sample_dims << "," << this->label_dims
                      << ", INSTEAD GOT " << sample.dimensions() << ","
                      << label.dimensions();
            throw std::invalid_argument(error_msg.str());
        }

        this->prebatch_samples.push_back(std::move(sample));
        this->prebatch_labels.push_back(std::move(label));
    }


    void Add(const std::vector<Tensor<SampleRank>> &samples,
             const std::vector<Tensor<LabelRank>> &labels)
    {
        if (samples.size() != labels.size()) {
            std::ostringstream error_msg;
            error_msg << "Dataset::Add GOT " << samples.size() << " SAMPLES BUT "
                      << labels.size() << " LABELS";

            throw std::invalid_argument(error_msg.str());
        }

        for (int i = 0; i < samples.size(); i++) {
            this->prebatch_samples.push_back(std::move(samples[i]));
            this->prebatch_labels.push_back(std::move(labels[i]));
        }
    }

#ifdef USING_OPENCV
    void Add(const std::string &image_path, const Tensor<LabelRank> &label)
    {
        if constexpr (SampleRank != 3) {
            std::ostringstream error_msg;
            error_msg << "Dataset::Add EXPECTS " << SampleRank
                      << " DIMENSIONAL INPUTS ";
            throw std::logic_error(error_msg.str());
        }

        Tensor<3> image_tensor;
        cv::Mat cv_matrix = cv::imread(image_path);

        if (cv_matrix.empty()) {
            throw std::invalid_argument(image_path + " is not a valid image file");
        }

        cv::cv2eigen(cv_matrix, image_tensor);
        this->Add(image_tensor, label);
    }
#endif

    void Shuffle()
    {
        std::random_device rand_device;
        std::mt19937_64 mt(rand_device());

        if (this->training_samples.size() != this->training_labels.size()) {
            std::ostringstream error_msg;
            error_msg << "Dataset::Add GOT " << this->training_samples.size()
                      << " SAMPLES BUT " << this->training_labels.size()
                      << " LABELS";

            throw std::invalid_argument(error_msg.str());
        }

        int dataset_size = this->training_samples.size() - 1;

        for (int i = 0; i < this->training_samples.size(); i++) {
            int random = std::uniform_int_distribution<int>(0, dataset_size)(mt);
            std::swap(this->training_samples[i], this->training_samples[random]);
            std::swap(this->training_labels[i], this->training_labels[random]);
        }
    }


    void Add(const std::string &path_to_csv,
             const std::vector<int> &label_col_indices,
             char delimiter = ',', bool ignore_mismatch = true)
    {
        // sort into descending order and remove uniques from vector
        std::vector<int> _label_col_indices = label_col_indices;
        std::sort(_label_col_indices.begin(), _label_col_indices.end(),
                  std::greater<>());
        _label_col_indices.erase(
                std::unique(_label_col_indices.begin(), _label_col_indices.end()),
                _label_col_indices.end());

        // string holds the current csv row as string
        std::string csv_row;
        std::ifstream csv(path_to_csv); // holds csv file in memory
        std::getline(csv, csv_row); // skip header

        // iterate through each csv row
        while (std::getline(csv, csv_row)) {
            std::vector<Scalar> sample;
            std::vector<Scalar> label;

            std::stringstream row_string(csv_row);
            std::string csv_col_value;
            int index = 0;

            // iterate through each csv value
            while (std::getline(row_string, csv_col_value, delimiter)) {
                std::vector<int> label_indices = _label_col_indices;

                // while iterating through each csv row value, if the current index is
                // one of the column label indices (whose vector was sorted descending),
                // add it to the label vector, else to the sample vector
                if (_label_col_indices.empty() || index != label_indices.back()) {
                    sample.push_back(
                            !csv_col_value.empty() ? std::stod(csv_col_value) : 0);
                    index++;
                }
                else {
                    label.push_back(std::stod(csv_col_value));
                    label_indices.pop_back();
                }
            }

            // check that the csv row has the expected amount of features
            if ((this->sample_dims.TotalSize() + this->label_dims.TotalSize()) !=
                sample.size() && !ignore_mismatch) {
                throw std::invalid_argument(std::string(
                        "Dataset::Add EXPECTED " +
                        std::to_string(this->sample_dims.TotalSize()) +
                        " FEATURES AND " +
                        std::to_string(this->label_dims.TotalSize()) +
                        " LABELS, GOT CSV ROW: " + csv_row));
            }

            this->prebatch_samples.push_back(
                    TensorMap<SampleRank>(sample.data(), this->sample_dims));
            this->prebatch_labels.push_back(
                    TensorMap<LabelRank>(label.data(), this->label_dims));
        }
    }


    static std::vector<std::vector<Scalar>>
    CSVToVector(const std::string &filename, char delimiter)
    {
        std::vector<std::vector<Scalar>> result;

        std::string csv_row;
        std::ifstream csv(filename);
        std::getline(csv, csv_row); // skip header

        // access each row
        while (std::getline(csv, csv_row)) {
            std::stringstream row_string(csv_row);
            std::string value;

            std::vector<Scalar> row;

            // access each value of the csv row and push to vector
            while (std::getline(row_string, value, delimiter)) {
                row.push_back(std::stod(value));
            }

            result.push_back(std::move(row));
        }

        return result;
    }

    std::vector<Tensor<SampleRank + 1>> training_samples;
    std::vector<Tensor<LabelRank + 1>> training_labels;

protected:
    void NormalizePreBatch()
    {
        for (Tensor<SampleRank> &tensor: this->prebatch_samples) {
            tensor = tensor / Tensor<0>(tensor.square().sum().sqrt()).coeff();
        }
    }


    void ShufflePreBatch()
    {
        std::random_device rand_device;
        std::mt19937_64 mt(rand_device());

        int dataset_size = this->prebatch_samples.size() - 1;

        for (int i = 0; i < this->prebatch_samples.size(); i++) {
            int random = std::uniform_int_distribution<int>(0, dataset_size)(mt);
            std::swap(this->prebatch_samples[i], this->prebatch_samples[random]);
            std::swap(this->prebatch_labels[i], this->prebatch_labels[random]);
        }
    }

    Dims<SampleRank> sample_dims;
    Dims<LabelRank> label_dims;

    std::vector<Tensor<SampleRank + 1>> test_samples;
    std::vector<Tensor<LabelRank + 1>> test_labels;

    std::vector<Tensor<SampleRank>> prebatch_samples;
    std::vector<Tensor<LabelRank>> prebatch_labels;
};

}

#endif //FLARE_DATASET_H
