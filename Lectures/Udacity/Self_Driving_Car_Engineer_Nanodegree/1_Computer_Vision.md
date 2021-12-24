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