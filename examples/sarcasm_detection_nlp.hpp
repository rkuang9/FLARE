//
// Created by R on 4/6/23.
//

#ifndef FLARE_SARCASM_DETECTION_NLP_HPP
#define FLARE_SARCASM_DETECTION_NLP_HPP

#include <iostream>
#include <flare/flare.hpp>
#include "lib/json-develop/single_include/nlohmann/json.hpp"


void SarcasmDetection()
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


    // add to dataset all headlines //excluding the last 10
    for (int i = 0; i < data.size() - 10; i++) {
        dataset.Add(dict.Sequence({data[i]["headline"].dump()})
                            .reshape(fl::Dims<1>(sentence_len)),
                    fl::Tensor<1>(1).setValues({{data[i]["is_sarcastic"]}}));
    }


    dataset.Batch(64, true);
    std::cout << "total headlines: " << data.size() << "\n"
              << "batched headlines: " << dataset.training_samples.size() << "\n";

    fl::Sequential model {
            new fl::Embedding(dict.Size(), 32, sentence_len),
            new fl::Bidirectional<fl::CONCAT>(new fl::LSTM(32, 16)),
            new fl::Dropout<2>(0.2),
            new fl::Dense<fl::ReLU>(32, 16, false),
            new fl::Dense<fl::Sigmoid>(16, 1, false),
    };

    fl::BinaryCrossEntropy<2> loss;
    fl::Adam opt;

    std::cout << "begin training\n";
    model.Fit(dataset.training_samples, dataset.training_labels, 15, loss, opt,
              {new fl::BinaryAccuracy<2>,
               new fl::Precision<2>,
               new fl::Recall<2>});

    for (auto i = data.size() - 40; i < data.size(); i++) {
        const std::string headline = data[i]["headline"].dump();

        fl::Tensor<2> sample = dict.Sequence({headline})
                .reshape(fl::Dims<2>(1, sentence_len));

        std::cout << "headline: " << headline << "\n";
        std::cout << "prediction: " << model.Predict<2>(sample) << "\n";
        std::cout << "label: " << data[i]["is_sarcastic"] << "\n\n";
    }

    std::cout << model.Predict<2>(dict.Sequence({"wealthy teen nearly experiences consequence"})) << "\n";
    std::cout << model.Predict<2>(dict.Sequence({"gallup poll rural whites prefer ahmadinejad to obama"})) << "\n";
    std::cout << model.Predict<2>(dict.Sequence({"kim jong un named the onions sexiest man alive for 2012"})) << "\n";
    std::cout << model.Predict<2>(dict.Sequence({"bush our long national nightmare of peace and prosperity is finally over"})) << "\n";
}


#endif //FLARE_SARCASM_DETECTION_NLP_HPP
