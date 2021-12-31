#### 2021/12/24
# Computer Vision
## Lesson 1: Introduction to Deep Learning for Computer Vision
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
---
#### 2021/12/28
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
---
#### 2021/12/29
## Lesson 2: The Machine Learning Workflow
### Introduction to the Machine Learning Workflow
- In the lesson, we are going to learn how to think about Machine Learning problems. Machine Learning (ML) is not only about cool math and modeling but also about choosing setting the problem, identifying the client needs and the long term goals. This lesson is going to be organized as follow:
  - We will practice framing Machine Learning problems by identifying the key stakeholders and choosing the correct metrics.
  - Because ML is about data, we will discuss the different challenges linked to data.
  - We will also tackle how to organize your dataset when solving a ML problem to be confident that you created a model that will perform well on new data.
  - Finally, we will see how you can leverage different tools to pinpoint your model's limitations.
- In this course, we will be using the German Traffic Sign Recognition Benchmark (GTSRB) multiple times for exercises. A downsampled version of the dataset has already been downloaded to your workspace.

### Big Picture
- In the following videos and lessons, we are going to take a deeper dive into each component of the workflow.
  - Problem setup is the phase where we set the boundaries of the problem and will be tackled in the next few videos.
  - The Data part of the workflow consists in getting familiar with the available dataset and will be the main focus of the next lesson on the camera sensor.
  - Modeling is such a critical step that we will spend 3 lessons on it. Modeling consists in choosing and training different models and picking the best one.
- Classifying Traffic Signs
  - Problem: Classifying traffic signs from images
  - Data: Thousand of images for each type of traffic sign
  - Model: Logistic regression, neural network

### Framing the Problem
- Do I even need Machine Learning?
- Who are the key stakeholders?
- What data do I have access to?
- Which metrics should I use?
- `As a machine learning engineer, it is easy to solely focus on model performances, but you may need to consider other factors as well`
- Unless you are taking part in a Machine Learning competition, the model's performance is rarely the only thing you care about. For example, in a self-driving car system, the model's inference time (the time it takes to provide a prediction) is also an important factor. A model that can digest 5 images per seconds is better than a model that can only manage one image per second, even if the second one is performing better. In this case, the inference time is also a metric to choose our model.

- Understanding your data pipeline is very important because it will drive your model development. In some cases, getting new data is relatively easy but annotating them (by associating a class name for example) may be expensive. In this case, you would want to create a model that requires less data or that can work with unlabeled data.
- `Machine Learning is an iterative process. You should always start with a simple model before building on complexity. Moreover, the business side often drives the metrics and the problem itself.`

### Identifying the Key Stakeholders
- Who is going to be impacted by the product?
- Stakeholders if ride sharing app:
  - Customers
  - Drivers
  - Engineering teams
- As a Machine Learning Engineer, you will rarely be the end user of your product. Therefore, you need to pinpoint the different stakeholders of the problem you are trying to solve. Why? Because this will drive your model development.
- Congratulations! Self-driving car technology will indeed impact many aspect of our society and this is what makes this technology so exciting. Daily commuters, insurance companies and environmentalists will benefit from the reduced traffic and car accidents. I recommend reading this excellent article on [the impact of self-driving cars on cities](https://www.washingtonpost.com/transportation/2019/07/20/city-planners-eye-self-driving-vehicles-correct-mistakes-th-century-auto/).

### Choosing Metrics
- Business metrics != Machine Learning metrics
- A good metric must be easy to understand and adapted to a specific problem
- Machine leanring metrics, like accuracy, may not be the best indicator of success from the business side
- You created an app that classify an image as containing a burger or not.
  - **True Positive (TP)**: The image contains a burger and the model predicts burger
  - **True Negative (TN)**: The image does not contain a burger and the model does not predict burger
  - **False Positive (FP)**: The image does not contain a burger and the model predicts burger
  - **False Negative (FN)**: The image contains a burger and the model does not predict burger

- Each Machine Learning problem requires its own metrics, and whereas some metrics like Accuracy may be suited for many problems, you need to keep in mind the consequences of misprediction. 
- Let's consider the following: you are building a spam classification algorithm. 
- Well, you should aim for very few False Positives, because you do not want your algorithm to classify some potentially important emails to your spam folder. 
- A False Negative however is simply a spam located in your inbox, which could be manually removed by the user.

### Classification & Object Detection Metrics
- **Precision = TP / (TP + FP)**
  - Of the elements classified as a particular class, how many did we get right? For example, we classified 6 images as containing burgers and only 5 of them actually contain a burger. The precision is 5/6
  - True 라고 한 것 중에 실제로 True인 비율
- **Recall = TP / (TP + FN)**
  - The number of images classified correctly divided by the total number of images. For example, we have 40 images of burgers and we classified 15 of them correctly. The recall is 15/40.   
  - True 여야 하는데 진짜 True라고 맞힌 비율
- **Accuracy = TP + TN / (TP + FN + FP + TN)**
  - (Only for classification problems) The number of correctly classified images over the total number of images.
--- 
- **Intersection over Union**
  - IOU is defined as the ratio of the intersection of bounding boxes and the union of bounding boxes.
  - `An IOU of 0.5 between a ground-truth bounding box and a detected bounding box is a pretty common threshold to qualify the detection as a TP.`
---
#### 2021/12/30
### Using a Desktop Workspace
- Certain exercises in this course make use of Workspaces attached to a virtual desktop that you can use to display visual output from your programs or to work with Linux desktop applications, such as Visual Studio Code.
#### Viewing the Desktop
- You can view the Desktop by clicking on the "Desktop" button in the lower right side of the Workspace. This will open a new window in your browser with the virtual desktop. The desktop has a Visual Studio Code application along with a terminal app called Terminator, which can be used to browse files or run code.

- Try the desktop now in the workspace below!

Note: Just as with other Workspace types, the Desktop will disconnect after 30 minutes of inactivity.
- Note that if you are working on exercises that make use of matplotlib to show plots or images, you can still choose to work directly out of the main workspace window when programming, but viewing the pop ups for any visualizations will require you to navigate to the Desktop button to view them.

#### Enabling GPU Mode in Desktop Workspaces
- Several desktop workspaces require the use of a GPU to display the desktop. If a desktop workspace has support for GPU, it will be displayed on the bottom left corner of the workspace, as shown in the snapshot below:
- You are provided with a fixed number of GPU hours at the beginning of this Nanodegree program. The amount of GPU time you have remaining in your account is displayed in the lower-left corner of your workspace.
- GPU Workspaces can also be run without time restrictions when the GPU mode is disabled. The "Enable"/"Disable" button can be used to toggle GPU mode. Note that in GPU-enabled workspaces, some libraries will not be available without GPU support.
  
NOTE: Toggling GPU support may switch the physical server your session connects to, which can cause data loss unless you click the save button before toggling GPU support.

#### 2021/12/31
### Data Acquisition and Visualization
#### Data
- Understand the data:
  - Orogin
  - Sensor
  - Labels
- ML engineering is all about data!
- `You will spend your time building data pipelines, creating data visualization, and trying to understand as much as possible about your dataset.`
- In many cases, you will need to gather your own data but in some, you will be able to leverage Open Source datasets, such as the Google Open Image Dataset. However, keep in mind the end goal and where your algorithm will be deployed or used.
- Because of something called domain gap, an algorithm trained on a specific dataset may not perform well on another. For example, a pedestrian detection algorithm trained on data gathered with a specific camera may not be able to accurately detect pedestrians on images captured with another camera.
