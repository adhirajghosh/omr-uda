### Music Object Detection in the Wild

The repo is based on [mmdetection](https://github.com/open-mmlab/mmdetection).

### Introduction
The past decade has witnessed significant progress on detecting objects in aerial images that are often distributed with large scale variations and arbitrary orientations. However most of existing methods rely on heuristically defined anchors with different scales, angles and aspect ratios and usually suffer from severe misalignment between anchor boxes and axis-aligned convolutional features, which leads to the common inconsistency between the classification score and localization accuracy. To address this issue, we propose a **Single-shot Alignment Network** (S<sup>2</sup>A-Net) consisting of two modules: a Feature Alignment Module (FAM) and an Oriented Detection Module (ODM). The FAM can generate high-quality anchors with an Anchor Refinement Network and adaptively align the convolutional features according to the corresponding anchor boxes with a novel Alignment Convolution. The ODM first adopts active rotating filters to encode the orientation information and then produces orientation-sensitive and orientation-invariant features to alleviate the inconsistency between classification score and localization accuracy. Besides, we further explore the approach to detect objects in large-size images, which leads to a better speed-accuracy trade-off. Extensive experiments demonstrate that our method can achieve state-of-the-art performance on DeepScores and IMSLP datasets while keeping high efficiency.

## Installation

Please refer to [install.md](docs/INSTALL.md) for installation and dataset preparation.


## Getting Started
Please see [getting_started.md](docs/GETTING_STARTED.md) for the basic usage of MMDetection.
