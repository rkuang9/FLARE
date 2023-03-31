#include <iostream>
#include <chrono>
#include <flare/flare.hpp>
#include "lib/json-develop/single_include/nlohmann/json.hpp"


void test()
{
    using json = nlohmann::json;
    std::ifstream f("sarcasm.json");
    json data = json::parse(f);
    f.close();

    int sentence_len = 30;
    int dictionary_size = 15000;

    fl::Tokenizer dict(dictionary_size, sentence_len);

    for (auto &headline: data) {
        dict.Add(headline["headline"].dump());
    }


    dict.Compile();

    fl::Dataset dataset(fl::Dims<1>(sentence_len), fl::Dims<1>(1));


    // add to dataset all headlines excluding the last 10
    for (int i = 0; i < data.size() - 10; i++) {
        dataset.Add(dict.Sequence({data[i]["headline"].dump()})
                            .reshape(fl::Dims<1>(sentence_len)),
                    fl::Tensor<1>(1).setValues({{data[i]["is_sarcastic"]}}));
    }


    dataset.Batch(64, false);
    std::cout << "samples: " << dataset.training_samples.size() << "\n";

    fl::Sequential model {
            new fl::Embedding(dict.Size(), 64, sentence_len),
            new fl::LSTM(64, 32),
            new fl::Dropout<2>(0.2),
            new fl::Dense<fl::ReLU>(32, 32, false),
            new fl::Dense<fl::Sigmoid>(32, 1, false),
    };

    fl::BinaryCrossEntropy<2> loss;
    fl::Adam opt;

    model.Fit(dataset.training_samples, dataset.training_labels, 20, loss, opt);

    for (auto i = data.size() - 20; i < data.size(); i++) {
        const std::string headline = data[i]["headline"].dump();

        fl::Tensor<2> sample = dict.Sequence({headline})
                .reshape(fl::Dims<2>(1, sentence_len));

        std::cout << "headline: " << headline << "\n";
        std::cout << "prediction: " << model.Predict<2>(sample) << "\n";
        std::cout << "label: " << data[i]["is_sarcastic"] << "\n\n";
    }
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    test();

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
            stop - start);
    std::cout << "\n\n" << "Run Time: " << time.count() / 1000.0 << " s";

    return 0;
}

// FLARE
// Fast Learning Architectures/Algorithms Ran Eagerly
// Fast Learning Architectures/Algorithms Rapid Execution
// Fast Learning Architectures/Algorithms Really Epic