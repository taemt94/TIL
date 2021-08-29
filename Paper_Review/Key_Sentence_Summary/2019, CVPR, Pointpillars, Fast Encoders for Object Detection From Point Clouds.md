# PointPillars : Fast Encoders for Object Detection From Point Clouds
### Authors : Alex H. Lang, et al.
### Year : 2019
### Journal : CVPR
### Citation : 594(2021.08. 기준)

## Abstract
- In this paper we consider the problem of encoding a point
cloud into a format appropriate for a downstream detection
pipeline.

- PointPillars, a novel encoder
which utilizes PointNets to learn a representation of
point clouds organized in vertical columns (pillars).
- While the encoded features can be used with any standard 2D convolutional
detection architecture, we further propose a lean downstream network.

## 1. Introduction
- To
detect and track moving objects, autonomous vehicles rely on several sensors
out of which the lidar is arguably the most important.
- Traditionally, a lidar robotics pipeline interprets
such point clouds as object detections through a bottomup
pipeline involving background subtraction, followed by
spatiotemporal clustering and classification [12, 9].
- Some early works focus on either using 3D convolutions [3] or a projection of the point cloud into the image
[14].
- Recent methods tend to view the lidar point cloud
from a bird’s eye view [2, 11, 31, 30].
- However, the bird’s eye view tends to be extremely
sparse which makes direct application of convolutional
neural networks impractical and inefficient.

- Building on the PointNet design developed by Qi et al. [22],
VoxelNet [31] was one of the first methods to truly do end-to-
end learning in this domain.
- VoxelNet divides the space
into voxels, applies a PointNet to each voxel, followed by
a 3D convolutional middle layer to consolidate the vertical
axis, after which a 2D convolutional detection architecture
is applied.
- However, VoxelNet's inference time, at 4:4 Hz, is too slow to deploy in real time.
- Recently SECOND [28] improved the inference speed of
VoxelNet but the 3D convolutions remain a bottleneck.
- In this work, we propose PointPillars: a method for object
detection in 3D that enables end-to-end learning with
only 2D convolutional layers.
- PointPillars uses a novel encoder
that learn features on pillars (vertical columns) of the
point cloud to predict 3D oriented boxes for objects.
- Advantages
  1. By learning
features instead of relying on fixed encoders, PointPillars
can leverage the full information represented by the point
cloud.
  2. By operating on pillars instead of voxels, there is no need to tune the binning of the vertical direction
by hand.
  3. Pillars are highly efficient because all
key operations can be formulated as 2D convolutions which
are extremely efficient to compute on a GPU.
  4. PointPillars requires no
hand-tuning to use different point cloud configurations.
- We evaluated our PointPillars network on the public
KITTI detection challenges which require detection of cars,
pedestrians, and cyclists in either the bird’s eye view (BEV)
or 3D [5].
- PointPillars dominates the current state of the
art including methods that use lidar and images.

### 1.1. Related Work
#### 1.1.1 Object detection using CNNs
- The
series of papers that followed [24, 7] advocate a two-stage
approach to Object detection problem in images.

- In a single-stage architecture a
dense set of anchor boxes is regressed and classified in a
single stage into a set of predictions providing a fast and
simple architecture.
- In this work, we use a single
stage method.

#### 1.1.2 Object detection in lidar point clouds
- Object detection in point clouds is an intrinsically three dimensional
problem.

- In the most common paradigm, the
point cloud is organized in voxels and the set of voxels in
each vertical column is encoded into a fixed-length, handcrafted,
feature encoding to form a pseudo-image which can
be processed by a standard image detection architecture.  
 -> MV3D, AVOD, PIXOR, Complex YOLO
- In their seminal work Qi et al. [22, 23] proposed a simple
architecture, PointNet, for learning from unordered point
sets, which offered a path to full end-to-end learning.
- VoxelNet
[31] is one of the first methods to deploy PointNets
for object detection in lidar point clouds.
- But like the earlier work that relied on 3D convolutions, VoxelNet is slow, requiring 225ms inference time (4:4 Hz) for a
single point cloud.
- Frustum PointNet [21], uses PointNets to segment and classify the point
cloud in a frustum generated from projecting a detection on
an image into 3D.
- But
its multi-stage design makes end-to-end learning impractical.
- Very recently SECOND [28] offered a series of improvements
to VoxelNet resulting in stronger performance
and a much improved speed of 20 Hz.
- However, they were
unable to remove the expensive 3D convolutional layers.

### 1.2. Contributions
1. We propose a novel point cloud encoder and network,
PointPillars, that operates on the point cloud to enable
end-to-end training of a 3D object detection network.
2. We show how all computations on pillars can be posed
as dense 2D convolutions which enables inference at
62 Hz; a factor of 2-4 times faster than other methods.
3. We conduct experiments on the KITTI dataset and
demonstrate state of the art results on cars, pedestrians,
and cyclists on both BEV and 3D benchmarks.
4. We conduct experiments on the KITTI dataset and
demonstrate state of the art results on cars, pedestrians,
and cyclists on both BEV and 3D benchmarks.

## 2. PointPillars Network
\!\[PointPillars network overview](./PointPillars_network_overview.JPG)
### 2.1. Pointcloud to Pseudo-Image