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
  - Convolution 2D Transpose
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
- [x] Softmax activation (12/1/2022)
- [x] Categorical cross entropy loss (12/1/2022)
- [x] Add progress bar to display during training (12/6/2022)
- [x] Multithreading support (12/12/2022)
- [x] Generative Adversarial Networks support (Conv2DTranspose valid padding 12/26/2022, BatchNormalization 1/13/2023, Conv2DTranspose output and same padding 1/15/2023)
- [x] Save and load weights from files
- [ ] Gated Recurrent Unit
- [ ] Long Short-Term Memory
- [ ] Bidirectional RNN
- [ ] Add bias support for layers
- [ ] Parse JSON datasets
- [ ] Evaluate a test set per epoch
- [ ] Accuracy metrics