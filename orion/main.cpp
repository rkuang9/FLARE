#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <sstream>
#include <fstream>

#include <orion/orion.hpp>


using namespace orion;


template<typename DataType>
std::vector<std::vector<DataType>>
CSVToVector(const std::string &filename, char delimiter = ',')
{
    std::vector<std::vector<DataType>> result;

    std::string csv_row;
    std::ifstream csv(filename);
    std::getline(csv, csv_row); // skip header

    // access each row
    while (std::getline(csv, csv_row)) {
        std::stringstream row_string(csv_row);
        std::string value;

        std::vector<DataType> row;

        // access each value of the csv row and push to vector
        while (std::getline(row_string, value, delimiter)) {
            row.push_back(std::stod(value));
        }

        result.push_back(std::move(row));
    }

    return result;
}


void DiabetesDetection()
{
    std::vector<std::vector<Scalar>> diabetes = CSVToVector<Scalar>("diabetes.csv");
    std::cout << "total samples: " << diabetes.size() << "\n\n";

    std::vector<std::vector<Scalar>> training_samples;
    std::vector<std::vector<Scalar>> training_labels;

    for (auto &row: diabetes) {
        training_labels.push_back({row.back()});
        row.pop_back(); // remove the 0/1 label, leaving only features
        training_samples.emplace_back(std::move(row));
    }

    Sequential model{
            new Dense<ReLU>(8, 12, false),
            new Dense<ReLU>(12, 8, false),
            new Dense<Sigmoid>(8, 1, false),
    };

    BinaryCrossEntropy loss;
    Adam opt(0.005);

    model.Compile(loss, opt);
    model.Fit(training_samples, training_labels, 90, 16);

    for (int i = 0; i < 10; i++) {
        std::cout << "predict: "
                << model.Predict(TensorMap<2>(training_samples[i].data(), 8, 1))
                << ", actual: " << training_labels[i].front() << "\n";
    }


}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    DiabetesDetection();

    auto stop = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\n" << "Run Time: " << ms.count() << " ms";

    return 0;
}