//
// Created by Macross on 3/13/23.
//

#ifndef FLARE_LEARN_ADDITION_HPP
#define FLARE_LEARN_ADDITION_HPP

#include <flare/flare.hpp>

void LearnAddition()
{
    fl::Dataset dataset(fl::Dims<1>(3), fl::Dims<1>(1));

    for (int i = 0; i < 1000; i++) {
        fl::Tensor<1> nums = fl::RandomUniform(fl::Dims<1>(3), 0, 1);
        fl::Tensor<1> sum = nums.sum().reshape(fl::Dims<1>(1));
        dataset.Add(nums, sum);
    }

    dataset.Batch(1, true);

    fl::Sequential model {
        new fl::Dense<fl::Linear>(3, 1, false),
    };

    fl::Tensor<2> test(1, 3);
    test.setValues({{1, 2, 3}});

    fl::MeanSquaredError<2> loss;
    fl::SGD opt;

    std::cout << "initial weights\n" << model.layers[0]->GetWeights2D().front() << "\n";
    std::cout << "sum: " << test << ": " << model.Predict<2>(test) << "\n\n";

    model.Fit(dataset.training_samples, dataset.training_labels, 2, loss, opt);

    std::cout << "trained weights\n" << model.layers[0]->GetWeights2D().front() << "\n";
    std::cout << "sum: " << test << ": " << model.Predict<2>(test) << "\n\n";
}

#endif //FLARE_LEARN_ADDITION_HPP
