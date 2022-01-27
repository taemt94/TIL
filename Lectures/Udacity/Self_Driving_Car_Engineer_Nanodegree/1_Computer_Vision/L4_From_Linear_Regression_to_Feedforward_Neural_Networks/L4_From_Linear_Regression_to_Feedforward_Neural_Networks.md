#### 2022/01/25
# Lesson 4: From Linear Regression to Feedforward Neural Networks
## Introduction to Machine Learning Algorithms
- In this course, we focus on the **prediction task**, which consists of predicting an outcome using a set of observations. In particular, we will focus on the following:

  - **Regression**: predict a quantity (eg, distance, time) and understand the relation between two variables.
  - **Classification**: assign a discrete class (eg, traffic sign) to an input.
- This lesson will be organized as follow:
  - Linear regression
  - Logistic regression
  - Optimization with gradient descent
  - Feed forward neural networks
  - Backpropagation
  - Image classification with a neural network

## Big Picture
- In this course, we will create ML **models** to predict an output based on an input observation. To estimate the fit of our model, a **loss function** will be used and the performances of the model created will be quantified with a chosen **metric**.

  - **Model**: a set of computations that extracts pattern from an input.
  - **Loss function / cost function**: A function that maps the outputs of a model to a single real number.
  - **Metric**: a function that quantifies the performances of an algorithm.
- Why do we need both the loss function and a metric?
  - First of all, a loss function must be differentiable, which is often not the case for metrics.
  - Secondly, the loss does not necessarily give us meaningful information on the model's performances, but rather describe how good a model is at fitting the data.

## Linear Regression
- A **linear regression** model is a type of ML algorithm that assumes a linear relation between the input variable and the output variable. Such a model is described by two parameters, the slope `m` and the intersection `b` such that `y = mx +b`. Fitting or training such a model requires to adjust `m` and `b` to minimize the chosen loss function.

#### 2021/01/26
## Linear Regression Loss Functions
- The **Mean Squared Error (MSE)** loss or **L2** loss is the one of the most common function used with linear regression algorithm. It is calculated by summing the square difference of the ground truths and the predictions. Because of the nature of the square function, this loss is very sensitive to **outliers** (out of distributions data points). 
- If your dataset contains many outliers, the **L1 (MAE)** loss (absolute difference between ground truth and predictions) may be a better candidate.

## Logistic Regression
- For classification problems, we can also use a linear expression to model the probability P(Y|X) of an input belonging to a certain category.
- Such a model is called **logistic regression** and looks like this: `P(Y|X) = mx+b`. 
- However, given that we want to model a probability, we need a way to constrain `mx+b` to the `[0, 1]` interval. 
### Logistic Function (Sigmoid)
- To do so, we are going to use the **logistic function (or sigmoid)**.  
    $sigmoid = \sigma(x) = \frac{e^x}{1 + e^x}$
- The logistic function maps any real number to the [0, 1] interval.
- Logistic Regression:  
  $f(X) = \frac{e^{mX+b}}{1 + e^{mX+b}}$
### Softmax Function
- **n classes**: X is a vector of n elements
- **Generalization** of the sigmoid function to **multiple dimensions**
- **Normarlize a vector X** into a **discrete probabilty distribution** (components add up to 1)
- The **softmax function** is the extension of the logistic function to multiple classes and takes a vector as input instead of a real number. 
- The softmax function outputs a discrete probability distribution: a vector of similar dimensions to the input but with all its components summing up to 1.  
  $Softmax = \sigma(x)_i = \frac{e^{x_i}}{\sum_{j=0}^ne^{x_j}}$
- Later in this lesson, we will describe the sigmoid and softmax functions as **activation functions**.

### Cross-Entropy Loss and One-Hot Encoding
- The **Cross Entropy (CE)** loss is the most common loss for classification problems. 
- The total loss is equal to the sum over all the observations of the dot product of the ground truth one-hot encoded vector and the log of the softmax probability vector.  
  $CE = L(y, \hat{y})=-\sum_{i=1}^ny_ilog(\hat{y_i})$

- For multiple classes classification problems, the ground truth labels need to be encoded as vectors to calculate. 
- A common approach is the **one-hot encoding** method, where each label of the dataset is assigned an integer. 
- This integer is used as the index of the only non zero element of the one-hot vector.

**Summary**: for classification problems, the labels need to be encoded to a vector of dimension `C`, where `C` is the number of classes in the dataset. Thanks to the **softmax function**, the model outputs a discrete probability distribution vector, also of dimension `C`. To calculate the **cross entropy loss** between the input and the output, we calculate the dot product of the **one hot vector** and the **log of the output**.

## Introduction To Tensorflow
- Tensorflow tensors are data structure sharing many attributes and properties with numpy arrays (such as .shape and broadcasting rules).
- They do have additional attributes, allowing the user to move tensors from one device to another (cpu to gpu for example).

#### 2021/01/27
## Gradient Descent
### Optimization
- Minimize the cost function to fing best weights  
  $\frac{dL}{dx}=0$
  - Fitting or training a ML algorithm consists of finding the combination of weights that **minimizes the loss function**. 
  - In some cases, it is possible to find an analytical solution (for example, linear regression with L2 loss).
- Analytical solution is not always available
  - However, for most of the algorithms tackled in this course, the analytical solution to the loss minimization problem does not exist.

### Local / Global Minima
- One of the challenges when using the gradient descent algorithm is the existence of **local minima**.
- Local minima are minima in a local subset of the loss function domain. 
- They are the smallest value this function can take, only in this small subset, as opposed to the **global minimum** where the loss function takes the smallest value of its entire domain. 
- The gradient descent algorithm can get stuck in local minima and outputs suboptimal solutions.
- Later in this lesson, we will see how other approaches solve this problem.

### Gradient Descent
- Iterative optimization algorithm
  - The **gradient descent** algorithm is an iterative approach to find the minimum of the loss function.
- Updates weight using the **gradient** and the **learning rate**
  - This algorithm takes a step towards this minimum using the gradient of the loss function scaled by a float called the **learning rate**.  
  $W=W-\alpha\frac{df}{dx}$
  - $\alpha$: learning rate

- Tensorflow **variable** are tensors with fixed type and shape but their value can be changed through operations. We need to use variables to calculate gradients in Tensorflow with the `tf.GradientTape` api.

## Stochastic Gradient Descent
###Mini-batch Gradient Descent
- Cannot optimize on the entire dataset at once
  - Because of memory limitations, the entire dataset is almost never loaded at once and fed through the model, as is the case in **batch gradient descent**.
- Batch gradient descent
  - Instead, **batches** of inputs are created.
- Overcome hardware limitations but needs additional processing
- Also called Stochastic Gradient descent (SGD)
  - Gradient descent performed on batches of **just one input at a time** is called **stochastic gradient descent (SGD)**, while **batches of more than one**, but not all at once (e.g. 20 batches of 200 images each), are called **mini-batch gradient descent**.