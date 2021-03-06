# TensorFlow: Predicting Political Party
## [Recurrent Neural Network using Python, TensorFlow & Keras]
<p float="left">
 <img src="Images/TFlow.png" width="175" />
</p>

I completed a machine-learning project to predict the political party of US Congress-members based on their voting record on 17-issues in the year 1984. Using the TensorFlow library and the keras API, we can predict the political party of a US congress-member with 95% accuracy. 

## Overview
### Dataset
The data is from ... The project was kept small (232, 17) to limit computing expense; larger data sets would require the use of GPU’s and distributed training.

### The Magic


### Model
Through experimentation I set Keras to use a Sequential model flow to allow for simple layer-building of five layers; Dense embedding layer, two hidden Dropout layers, an output layer with a binary-classification sigmoid function, and the final compilation layer. I used the scikit-learn Python library to make cross-validation easier.  *Relu* is used as the activiation function in the first three layers; and *sigmoid* as the activiation function in the output layer (because the output is a binary classification).  The loss-function in the compilation layer is *binary cross-entropy* and *adam* is used as the optimizer in this layer.  I wrapped the keras classifier in scikit-learn.<br>
**Below: Python Code for Model Layers and Settings:**
<p float="left">
 <img src="Images/PartyCode.png" width="600" />
</p>

### Results
My experimentation with Keras settings improved the model’s accuracy from 94% to 95%.

### Additional Learning Topics
1. Explore TensorFlow settings (hyper-parameter tuning)
2. Use larger datasets on cloud server resources - big data use of TensorFlow
3. Consider transitioning to PyTorch

## About TensorFlow
TensorFlow is an open-source library for machine learning, specifically focused on training and inference of deep neural networks; it is perfectly aligned with working in the *Deep Learning* space.  Keras is an API built on top of Tensor Flow 2.0.  Keras minimizes the number of user-actions required for use cases and enables fast experimentation.  TensorFlow is often used to train neural networks (NN), to build recommender systems, and to train generative advesarial networs (GANNs).  TensorFlow can train and run deep neural networks for handwritten digit classification, image recognition, word embedding and creation of various sequence models.<br>
The base unit, the *Tensor* is a multi-dimensional array of data. In math, tensors are described as geometric objects that describe linear relationships between other geometric objects. More formally, Tensorflow is a symbolic math library based on dataflow and differentiable programming.<br>
TensorFlow Models are very *computationally expensive*, often requiring the use of a GPU or - even better than a GPU - a *tensor processing unit (TPU)*. Tensors are identified by the following three parameters: *Rank, Shape* and *Type*.<br>

Tensor flow is designed to function in two main deep neural networks:
1. Convolutional Neural Network, and
2. Recurrent Neural Network

**Keras & TensorFlow** <br>
Keras is compact, easy to learn, high-level Python library run on top of TensorFlow framework. It is made with focus of understanding deep learning techniques, such as creating layers for neural networks maintaining the concepts of shapes and mathematical details. 

**Tensors & Graphs**<br>
The core component of TensorFlow is the compuational graph and Tensors which traverse among all the nodes through what is called *edges*.
Mathematically, a **Tensor** is a N-dimensional vector, meaning it can be used to represent N-dimensional datasets.  The **Flow** refers to a computational graph, and each computation in TensorFlow is represented as a Data Flow Graph.  Each node in the graph represents a mathematical operation like addition, subtraction etc. And each operation results in the formation of new Tensor.<br>
<p align="center">
 <b>Below: Tensors Rank & Dimension and Computational Graph (Flow)</b>
</p>
<p align="center">
  <img src="Images/tensorranks.png" width="400" />
  <img src="Images/flow.png" width="350" />
</p>
<br>
<br>

**Tensor Flow Playground**
The best way to understand a Neural Network (and how TensorFlow operates on a NN) is throught *Visualization*.  The Neural Network Playground allows anyone to experiment with four datasets, and to create the entire NN model, then visually watch Tensor Flow converge on a solution.

TRY IT: https://playground.tensorflow.org
<p float="left">
 <img src="Images/TFPlaygound.png" width="600" />
</p>



