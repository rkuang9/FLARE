//
// Created by Macross on 12/6/22.
//

#ifndef ORION_HEART_ATTACK_PREDICTION_H
#define ORION_HEART_ATTACK_PREDICTION_H

#include <orion/orion.hpp>


// Working example, requires fine-tuning but passes gradient check (unless gradients vanish)
// At only 303 samples and labels, it's possible that this dataset is too small
// heart attack dataset from:
// https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
void HeartAttackPrediction()
{
    using namespace orion;

    Dataset dataset(Dims<1>(13), Dims<1>(1));
    dataset.Add("heart.csv", std::vector<int> {13});
    dataset.Batch(1, true, false);

    std::cout << "total samples: " << dataset.training_samples.size() << "\n";

    Sequential model {
            new Dense<ReLU>(13, 256, false),
            new Dense<Sigmoid>(256, 1, false),
    };

    MeanSquaredError<2> loss;
    Adam opt;

    model.Fit(dataset.training_samples, dataset.training_labels, 14, loss, opt);

    for (int i = 0; i < dataset.training_samples.size(); i++) {
        std::cout << "predict " << model.Predict<2>(dataset.training_samples[i])
                  << ", label " << dataset.training_labels[i] << "\n";
    }

    std::cout << "gradient check: "
              << model.GradientCheck(dataset.training_samples.front(),
                                     dataset.training_labels.front(), loss);
}


#endif //ORION_HEART_ATTACK_PREDICTION_H
