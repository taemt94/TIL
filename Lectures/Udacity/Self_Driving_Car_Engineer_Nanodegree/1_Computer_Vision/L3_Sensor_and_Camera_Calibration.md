#### 2022/01/18
## Lesson 3: Sensor and Camera Calibration
### Intro to the Camera Sensor
- Understand the data = Understand the sensor
- Self driving cars have multiple cameras
- Raw data needs to be processed before being used by a ML algorithm
- This lesson will be organized as follow:
  - The camera sensor and its distortion effect
  - The camera pinhole model
  - Camera calibration
  - RGB and other color systems
  - Image manipulation in Python
- Introducing Cezanne
  - Expert in Computer Vision
  - Masters in Electrical Engineering from Stanford University
  - Former Researcher in Genomics and Biomedical Imaging
  - You'll learn about:
    - Distortion correction
    - Camera pinhole model
    - How to use OpenCV to calibrate cameras

### Big Picture
- Camera Usages in Self Driving Car
  - **High resolution sensor**
  - Colors / Optical character recognition
  - **Depth reconstruction** with stereo cameras
  - Cost / space efficient
- Camera Limitations
  - Sensitive to **weather**
  - Information needs to be **extracted** with an algorithm
  - Not great for **depth estimation**
- Cameras are optical instruments capturing the **light intensity** on a digital image. The most important characteristics of a camera for a ML engineer are the following:

- **Resolution**: Number of pixels the image captured by the camera is made of (usually described in mega pixels).
- **Aperture**: size of the opening where the light enters the camera. Controls the amount of light received by the sensor.
- **Shutter speed**: duration that the sensor is exposed to the light. Also controls the amount of light by the sensor.
- **Focal length / field of view**: this parameter controls the angle of view of the image.

### Distortion Correction
#### Distortion
- Image distortion occurs when a camera looks at 3D objects in the real world and transforms them into a 2D image; this transformation isn’t perfect. Distortion actually changes what the shape and size of these 3D objects appear to be. So, the first step in analyzing camera images, is to undo this distortion so that you can get correct and useful information out of them.
- Helpful Quiz : Why is it important to correct for image distortion?
  - Distortion can change the apparent size of an object in an image.
  - Distortion can change the apparent shape of an object in an image.
  - Distortion can cause an object's appearance to change depending on where it is in the field of view.
  - Distortion can make objects appear closer or farther away than they actually are.