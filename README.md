# OrionNN
A neural network library in active development written in C++ and based on Eigen Tensor
with an interface similar to TensorFlow.

I started this project to learn more about artificial intelligence and because there aren't
many open-source fully featured and easy to understand neural network libraries.

Contribution is more than welcome, please email rkuang25@gmail.com.


# Features
- Layers 
  - Dense
  - Embedding (temporarily not working)
  - Global Average Pooling 1D (temporarily not working)
  - Convolution 2D
  - Max Pooling 2D
  - Flatten
  - Convolution 2D Transpose (currently only supports valid padding)
  - Batch Normalization
- Activation Functions
  - Sigmoid
  - Hyperbolic tangent (TanH)
  - Rectified Linear Unit (ReLU)
  - Linear
  - Swish
  - Softplus
  - Softsign
  - Scaled Exponential Linear Unit (SELU)
  - Exponential Linear Unit (ELU)
  - Softmax (backpropagation supported for Dense layers only)
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
  - Loss (recorded by loss function)
- Dataset Class
  - Batching
  - Shuffling
  - Normalization
  - Image-To-Tensor (requires OpenCV)


# TODO
- Softmax activation (done 12/1/2022)
- Categorical cross entropy loss (done 12/1/2022)
- Add progress bar to display during training (done 12/6/2022)
- Multithreading support (partially done 12/12/2022, only 2 threads)
- Generative Adversarial Networks for generating images (done, added Conv2DTranspose valid padding only 12/26/2022, added BatchNormalization 1/13/2023)
- Save and load models from files
- Add bias support for layers
- Parse JSON datasets
- Evaluate a test set per epoch
- Accuracy metrics