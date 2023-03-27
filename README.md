# FLARE
#### Fast Learning Algorithms Ran Eagerly

A neural network library in active development, written in C++, built on Eigen Tensor, and designed
with an interface similar to TensorFlow. The initial goal is to learn and  implement common neural network
architectures found in computer vision and natural language processing.

Contribute and learn AI together.

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
  - Dropout
  - Gated Recurrent Unit
  - Long Short-Term Memory
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
- [ ] Bidirectional RNN
- [ ] Add bias support for layers
- [ ] Parse JSON datasets
- [ ] Evaluate a test set per epoch
- [ ] Accuracy metrics