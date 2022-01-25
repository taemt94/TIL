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