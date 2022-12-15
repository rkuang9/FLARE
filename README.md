# OrionNN
A neural network library in active development written in C++ and based on Eigen Tensor.
This library intends to provide an interface and performance similar to TensorFlow 2.

# Features
- Layers 
  - Dense
  - Convolution 2D
  - Max Pooling 2D
  - Flatten
  - Embedding (temporarily not working)
  - Global Average Pooling 1D (temporarily not working)
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
  - Softmax (only supported for Dense layers for backpropagation)
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
- Support for Generative Adversarial Networks
- Add bias support for layers
- Parse JSON datasets
- Evaluate a test set per epoch
- Accuracy metrics