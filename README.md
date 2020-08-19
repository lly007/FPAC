# Frame Point Attentions Convolution for Point Cloud Classification(FPAC)

**FPAC is a spatial convolution operator for point cloud**


![The Overview of FPAC](https://github.com/lly007/FPAC/blob/master/image/fig2.png?raw=true "The Overview of FPAC")


This is an implementation of FPAC using tensorflow and keras.


# Introduction

This project propose a new scheme, called Frame Point Attention Convolution (FPAC), for performing the 3D point cloud convolution and extracting the features from the individual cloud points. 





#  Environment
This project passed the test in the following environment
### Software
- Microsoft Windows 10 1904
- TensorFlow 2.1
- NVIDIA CUDA Toolkit 10.2
- NVIDIA cuDNN 7.6

### Harware
- Intel Core i9 9900K
- NVIDIA TITAN RTX
- 64GB RAM



# Details
- **FPAC.py** is the keras model of FPAC, you may use this to build your own network.
- **train.py** is an example of classification training.
- **PointPooling.py** is the max pooling layer for point clouds.
- **PrepDataAndFPS.py** is used to preprocess the ModelNet data set, and perform FPS downsampling on the model in ModelNet to get the index of the sampling point. This process will take a long time.

# ModelNet40
You may get the ModelNet40 Dataset from [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip).

