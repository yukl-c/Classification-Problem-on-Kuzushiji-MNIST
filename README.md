# Classification-Problem-on-Kuzushiji-MNIST
Classification Problem on Kuzushiji-MNIST with PyTorch

# Objective
To address an image classification system to classify the words in Kuzushiji-MNIST (KMNIST) dataset with convolution neural network as base. Then to compare with algorithms, loss functions, learning rates, batch sizes.

# Background
- Kuzushiji: A kind of Japanese cursive writing with long history, gradually declined. Only professionals received Kuzushiji education can read. With not this kind of experts, it is much difficult to read these huge amounts of Japanese cultural work.
- CNN: 
 * A type of deep learning model specifically designed for processing structured grid-like data, such as images and videos. Its neuron-like elements can respond to surrounding units within a portion of the coverage area.
 * convolution: features of each block are taken form original images and different filter matrix -> the input to the next layer
 * Pooling: reduce the parameter need to be trained, avoid overfitting, retain the main feature of images even zooming out
 * Reasons choosing CNN: 
   - higher accuracy by sharing features parameter and reduce dimension
   - performs better than multilayer preception to understand spatial between pixels of images better for image complication better

# KMNIST Dataset: 
- a basic dataset similar to MNIST
- focus on cursive Kuzushiji
- 70000 images 
- 10 classes: ‘o’, ‘ki’, ‘su’, ‘tsu’, ‘na’, ‘ha’, ‘ma’, ‘ya’, ‘re’, ‘wo’

# Modelling
- 1st convolution layer:
 input: 1 channel; output: 6 layers; kernel size: 5*5; padding: 2 pixels; activation function: ReLU
- 1st pooling layer: 
 kernel size:2*2;  stride:2 pixels
- 2nd convolution layer: 
input: 6 channel; output: 16 layers; kernel size: 5*5; padding: 2 pixels; activation function: ReLU
- 2nd pooling layer: same 1st pooling layer
- flattern layer: form 1-D tensor from matrix as input after convolution and pooling to make the judgement stable
- Output: dimension matches the number of classes

# Training
- train a model with a GPU
- generate a graph and animation of changes of training and testing accuracy, training loss per epoch
- With no forwards or backwards gradient calculation through stochastic gradient descent, timer stop.
- show the finalized result of training and testing accuracy, loss and number of examples per cpu
- baseline setting:
 * loss function: cross entropy(CE)
 *  optimizer:  Stochastic gradient descent (SDG)
 *   learning rate: 0.1
 * batch size: 256
 * no. of epoch: 10

# Evaluation 
- Metrics: loss & test accuracy(acc)
- Baseline(CNN, CE, lr:0.1, batch size: 256): loss: 0.094; test acc: 0.913
- Comparing citerias:
  loss function: multi margin error(MME)
  leraning rate: 0.001
  batch size: 32
  algorithm: softmax
  Result(algorithm, loss functin, learning rate, batch size):
  1. (CNN, MME, lr:0.1, batch size: 256): loss: 0.021; test acc: 0.878
  2. (CNN, CE, lr:0.001, batch size: 256): loss: 1.724; test acc: 0.448
  3. (CNN, MME, lr:0.001, batch size: 256): loss: 0.389; test acc: 0.439
  4. (CNN, CE, lr:0.1, batch size: 32): loss: 0.024; test acc: 0.933
  5. (CNN, MME, lr:0.1, batch size: 32): loss: 0.012; test acc: 0.899
  6. (CNN, CE, lr:0.001, batch size: 32): loss: 0.834; test acc: 0.621
  7. (CNN, MME, lr:0.001, batch size: 32): loss: 0.151; test acc: 0.581
  8. (softmax, CE, lr:0.1, batch size: 256): loss: 0.566; test acc: 0.704
  9. (softmax, CE, lr:0.001, batch size: 256): loss: 0.566; test acc: 0.704
  10. (softmax, CE, lr:0.1, batch size: 32): loss: 0.590; test acc: 0.696
  11. (softmax, CE, lr:0.001, batch size: 32): loss: 0.777; test acc: 0.639
    
test acc >0.9: baseline, 5. (CNN, CE, lr:0.1, batch size: 32). These 2 models perform high accuracy. 
While loss: 5. (CNN, CE, lr:0.1, batch size: 32) > baseline. 
Thus, (CNN, CE, lr:0.1, batch size: 32) is the most suitable model.

# Limitations:
* mismatching of words with similar strockes of words that reduce result accruacy
e.g. ha’(は) and ‘su’(す); ‘tsu’(つ) and ‘ya’(や)
- mismatching of words due to the different standard of images. It leads to higher chance mismatching as other words and reduce accruacy
* some words: write in different style of writing, poor image quality

# Improvemnt
- enlarge dataset size
- choose the data with clear image




