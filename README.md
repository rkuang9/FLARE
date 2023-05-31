# FLARE
#### Fast Learning Algorithms Ran Eagerly

FLARE is a lightweight header-only C++ machine learning library
featuring popular neural network architectures. Its simple interface
allows developers to implement code for natural language processing
tasks and image recognition.

# How To Use
#### FLARE has the following dependencies
- [Eigen 3](https://gitlab.com/libeigen/eigen)


Place FLARE and Eigen into your project folder and add the following to your CMakeLists.txt file
```
include_directories(FLARE)
include_directories(Eigen)
```

- compile with the following flags for better performance
  - O2
  - fopenmp (only if OpenMP is available)

```#include <flare/flare.hpp> ``` to start using FLARE


## Example
Train a model to add three numbers
```cpp
#include <iostream>
#include <flare/flare.hpp>

int main()
{
    using namespace fl;

    Dataset dataset(Dims<1>(3), Dims<1>(1));

    for (int i = 0; i < 1000; i++) {
        Tensor<1> nums = RandomUniform(Dims<1>(3), 0, 1);
        Tensor<1> sum = nums.sum().reshape(Dims<1>(1));
        dataset.Add(nums, sum);
    }

    dataset.Batch(1, true);

    Sequential model {
            new Dense<Linear>(3, 1, false),
    };

    MeanSquaredError<2> loss;
    SGD opt;

    Tensor<2> test(1, 3);
    test.setValues({{1, 2, 3}});

    std::cout << "initial weights\n" << model.layers[0]->GetWeights2D()[0] << "\n";
    std::cout << "sum: " << test << ": " << model.Predict<2>(test) << "\n\n";

    model.Fit(dataset.training_samples, dataset.training_labels, 2, loss, opt);

    std::cout << "trained weights\n" << model.layers[0]->GetWeights2D()[0] << "\n";
    std::cout << "sum: " << test << ": " << model.Predict<2>(test) << "\n\n";
}
```

---

# Features
- Layers 
  - Dense
  - Embedding
  - Convolution 2D
  - Max Pooling 2D
  - Flatten
  - Convolution 2D Transpose
  - Batch Normalization
  - Dropout
  - Gated Recurrent Unit
  - Long Short-Term Memory
  - Bidirectional RNN
- Activation Functions
  - Sigmoid
  - TanH
  - ReLU
  - Linear
  - Swish
  - Softplus
  - Softsign
  - SELU
  - ELU
  - Softmax (rank 2 tensor layer outputs only)
- Loss
  - Binary Cross-entropy
  - Mean Squared Error
  - Categorical Cross-entropy (also known as Cross-entropy)
  - Mean Absolute Error
  - Kullback-Leibler Divergence
- Optimizer
  - Stochastic Gradient Descent (with momentum)
  - Adam  
  - RMSprop
- Weight Initializers
  - Glorot Uniform
  - Glorot Normal
  - He Uniform
  - He Normal
  - Lecun Uniform
  - Lecun Normal
- Metrics
  - Loss
- Dataset Class
  - Batching
  - Shuffling
  - Normalization
  - Image-To-Tensor (requires OpenCV)


# TODO
- [x] Softmax activation (12/1/2022)
- [x] Categorical cross entropy loss (12/1/2022)
- [x] Add progress bar to display during training (12/6/2022)
- [x] Multithreading support (12/12/2022)
- [x] Generative Adversarial Networks support (Conv2DTranspose valid padding 12/26/2022, BatchNormalization 1/13/2023, Conv2DTranspose output and same padding 1/15/2023)
- [x] Save and load weights from files
- [x] Gated Recurrent Unit (3/9/2023)
- [x] Long Short-Term Memory (3/26/2023)
- [x] Bidirectional RNN (4/5/2023)
- [ ] Add bias support for layers
- [ ] Evaluate a test set per epoch
- [ ] Accuracy metrics
- [ ] Attention architecture