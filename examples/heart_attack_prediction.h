//
// Created by Macross on 12/6/22.
//

#ifndef ORION_HEART_ATTACK_PREDICTION_H
#define ORION_HEART_ATTACK_PREDICTION_H

#include <orion/orion.hpp>

// working example, requires fine-tuning but passes gradient check (unless gradients vanish)
// heart attack dataset from:
// https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
void HeartAttackPrediction()
{
    using namespace orion;

    Dataset dataset(Dims<1>(13), Dims<1>(1));
    dataset.Add("heart.csv", std::vector<int> {13});
    dataset.Batch(1, true, false);

    Sequential model {
            new Dense<ReLU>(13, 256, false),
            new Dense<Sigmoid>(256, 1, false),
    };

    BinaryCrossEntropy loss;
    Adam opt;

    model.Compile(loss, opt);
    model.Fit(dataset.training_samples, dataset.training_labels, 15);

    for (int i = 0; i < dataset.training_samples.size(); i++) {
        std::cout << "predict " << model.Predict<2>(dataset.training_samples[i])
                  << ", label " << dataset.training_labels[i] << "\n";
    }
}

#endif //ORION_HEART_ATTACK_PREDICTION_H
