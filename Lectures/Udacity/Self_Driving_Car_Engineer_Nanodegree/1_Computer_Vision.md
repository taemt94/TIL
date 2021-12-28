## 2021/12/24
### Course Outline
- Introduction to Deep Learning for Computer Vision (this lesson)
- Overview of the Machine Learning Workflow
- Linear and Logistic Regression: Introduction to Neural Networks
- Classify Images using a Convolutional Neural Network
- Detect Objects in an Image
- Final Project

### Key Stakeholders
- Impacts on society
  - Improving commute experience
  - Reducing traffic
  - Reducing number of accidents
  - Changing cities layouts
  - Reducing air pollution

- Engineering team involved
  - Operations team : data acquisition, labeling, mapping
  - Hareware team : sensor (Lidar, cameras) and embedded system
  - Data engineering team : data pipelines
  - R&D team : algorithm development

### Introduction to Deep Learning and Computer Vision
- Artificial Intelligence (AI): a system that leverages information from its environment to make decisions. For example, a video game bot.
- Machine Learning (ML): an AI that does not need to be explicitly programmed, and instead learns from data. For example, a spam classification algorithm.
- Deep Learning (DL): a subset of ML algorithms that do not require handcrafted features and can work with raw data. For example, an object detection algorithm with a convolutional neural network.

- Supervised Learning
  - Use annotated data to train an algorithm
  - Input (variable) / $X$ / Observation: the input data to the algorithm. For example, in spam classification, it would be an email.
  - Ground truth / $Y$ / label: the known associated label with the input data. For example, a human created label describing the email as a spam or not.
  - Output variable / $\hat{Y}$ / prediction: the model prediction given the input data. For example, the model predicts the input as being spam.

## 2021/12/28
### Deep Learning for Computer Vision
- Deep learning excels at analyzing digital images
  - Better performances than traditional Coputer vision methods
  - Ability to understand complex environments
- It comes with a cost:
  - Slower
  - Harder to deploy
- Deep Learning algorithms are now the state of the art (SOTA) in most computer vision problems, such as image classification or object detection. Therefore, they are now in every SDC system. However, using deep learning adds additional constraints to the system because they require more computational power.

### History of Deep Learning
- Artificial neural networks (ANN) or simply neural networks are the type of systems at the core of deep learning algorithms.
- ANN: machine learning algorithms vaguely based on human neural networks.
- Neurons: the basic unit of neural networks. Takes an input signal and is activated or not based on the input value and the neuron's weights.
- Layer: structure containing multiple neurons. Layers are stacked to create a neural network.
- Deep Learning before 2012
  - Multilayer neural network developed in 1970s.
  - 1989: Backpropagation algorithm and first neural network was created by Yann LeCun to detet handwritten digits.
  - 1990s: winter of deep learning.
  - Early 2000s, neural network start to be adopted in the industry due to the development of hardware.
  - 2009: ImageNet
  - 2012: GPUs and AlexNet
  - 2020: Deep Learning is everywhere

### Introduction to Tensorflow
- In this course, we will be using the TensorFlow library to create our machine learning models. TensorFlow is one of the most popular ML libraries and is used by many companies to develop and train algorithms. TensorFlow makes it very easy for the user to deploy such algorithms on different platforms, from a smartphone device to the cloud.
- Tensorflow
  - Machine Learning library making easy to develop and train ML models
  - Python library using C++ fundations
  - In addition:
    - Deployment to web browser
    - Deployment to mobile devices
    - Deployment to the cloud

### Register for the Waymo Open Dataset
- One of the truly exciting parts of this version of the Self-Driving Car Engineer Nanodegree program is the usage of the Waymo Open Dataset in some of the exercises and projects. Formerly the Google Self-Driving Car project (also originally headed by Sebastian Thrun, Udacity's founder, years ago), Waymo is one of the leaders of self-driving car technology. The Waymo Open Dataset contains tons of high quality data from both lidar and camera sensors from diverse locations and conditions.
- Before we continue in the course, it's important to go ahead and register for the Waymo Open Dataset, making sure to put your Institution as "Udacity" using this link. As it may take up to 48 hours for your request to be approved, it's important to get it done now so you are able to more easily complete any related exercises and projects.

### Tools, Environment & Dependencies
- In this course, you will need the following:

  - Install gsutil: a Python application to manipulate Google Cloud Storage items. You will find the tutorial to install it here.
  - Create a Github account: a version control system. You will need to create an account here. You will need a github account to access some of the material and create your submission for the final project. If you already have an account, you are good to go for this step!
  - Set up an Integrated development environment (IDE): a software application to write code. For this course, I would recommend either Pycharm or VS Code.

### Project: Object Detection in an Urban Environment
1. Train an object detection model on the Waymo dataset.
2. Learn how to use the Tensorflow object detection API.
3. Learn how to pick the best parameters for your model.
4. Perform an in-depth error analysis to understand your model's limitations.

- Github repository to add to your portfolio.
- Video to demo your model's performance.