---
title: 6D Pose Estimation Paper Review (2)                                
key: The-6D-Pose-Estimation-Paper-Review-2
tags:
- 6D Pose Estimation
mode: immersive
mathjax: true
header:
  theme: dark
article_header:
  type: overlay
  theme: dark
  background_color: '#000000'   
  background_image:
    gradient: 'linear-gradient(135deg, rgba(34, 139, 87 , .1), rgba(139, 34, 139, .1))'
    src: assets/images/cover5.jpg
---
<!--#203028-->
This is the second blog of the "6D pose estimation paper review" series. Here I  write a summary of the paper "DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion" from CVPR 2019. In addition, it is one of my assignments of the course "Visual Media Engineering" at the University of Tokyo.                  

## Introduction:
This paper introduces a generic framework for estimating 6D pose of a set of known objects from RGB-D images, namely DenseFusion. This framework firstly processes the two data sources individually and uses a novel dense fusion network to extract pixel-wise dense feature embedding, from which the pose is estimated. After that,  an end-to-end iterative pose refinement procedure is applied to further improve the pose estimation while achieving near real-time inference. 

This method outperforms the state-of-the-art PoseCNN after ICP refinement by 3.5% in pose accuracy while being 200x faster in inference time
<div style="width:50%; margin:0 auto;" align="center" markdown="1">
![robot](/assets/images/DenseFusion_robot.png "DenseDusion_robot")
</div>

## Model:

A 6D pose is composed by a rotation $$R$$ ∈ SO(3) and a translation $$t$$ ∈ $R^3$, p = $$[R \lvert t]$$. The poses are defined with respect to the camera coordinate frame.



<!--![CNN](/assets/images/Seamless_CNN.png){:width="650px"}-->

<!--img class="image image--xl" src="/images/Seamless_CNN.png"/-->
<div style="width:100%; margin:0 auto;" align="center" markdown="1">
![formula](/assets/images/DenseFusion_model.png " model")
</div>

Figure 2 represents the CNN architecture of DenseFusion. Based on this model, I will explain each stage step by step.
### Semantic Segmentation

The first stage is **Semantic Segmentation**, which takes color image as input and performs semantic segmentation for each known object category. An N+1-channelled semantic segmentation map is generated. Each channel is a binary mask where active pixels depict objects of each of the N possible known classes. Then, for each segmented object, the masked depth pixels (converted to 3D point cloud) as well as an image patch cropped by the bounding box of the mask are fed to the second stage.

###  Dense Feature Extraction

This stage, namely **Dense Feature Extraction**, processes the results of the segmentation and prepares for the estimation the object’s 6D pose.
 
- Dense color image feature embedding

 A fully convolutional network processes the color information and maps each pixel in the image crop to a color feature embedding. The image embedding network is a CNN-based encoder-decoder architecture that maps an image of size H × W × 3 into a $$H \times W \times d_{rgb}$$ embedding space. Each pixel of the embedding is a $$d_{rgb}$$-dimensional vector representing the appearance information of the input image at the corresponding location.


- Dense 3D point cloud feature embedding

 "Dense 3D point cloud feature embedding" first converts the segmented depth pixels into a 3D point cloud using the known camera intrinsics, and then uses a PointNet-like architecture to extract geometric features. In detail, the geometric embedding network generates a dense per-point feature by mapping each of the P segmented points to a $$d_{geo}$$-dimensional feature space.  The PointNet-like architecture  uses average-pooling as opposed to the commonly used max-pooling as the symmetric reduction function.


### Pixel-wise Dense Fusion

This is a novel pixel-wise dense fusion network that effectively combines the extracted features, especially for pose estimation under heavy occlusion and imperfect segmentation. The pexel-wise dense fusion performs local per-pixel fusion instead of global fusion so that we can make predictions based on each fused feature. In this way, we can potentially select the predictions based on the visible part of the object and minimize the effects of occlusion and segmentation noise. In detail, it first associates the geometric feature of each point to its corresponding image feature pixel based on a projection onto the image plane using the known camera intrinsic parameters. The obtained pairs of features are then concatenated and fed to another network to generate a fixed-size global feature vector using a symmetric reduction function. 
We feed each of the resulting per-pixel features into a final network that predicts the object’s 6D pose. The result is a set of P predicted poses, one per feature.
Moveover, in order to train the network to select the most likely pose estimation,  the network  outputs a confidence score $$c_i$$ for each prediction in addition to the pose estimation predictions. 

### 6D Object Pose Estimation

In this paper, the pose estimation loss is defined as the distance between the points sampled on the objects model in ground truth pose and corresponding points on the same model transformed by the predicted pose. Specifically, the loss to minimize for the prediction per dense-pixel is defined as 

$$
L_{i}^{p}=\frac{1}{M} \sum_{j}\left\|\left(R x_{j}+t\right)-\left(\hat{R}_{i} x_{j}+\hat{t}_{i}\right)\right\|
$$

where $$x_j$$ denotes the jth point of the M randomly selected 3D points from the object’s 3D model, p = $$[R \lvert t]$$ is the ground truth pose, and $$\hat{p}_{i}=\left[\hat{R}_{i} \mid \hat{t}_{i}\right]$$ is the predicted pose generated from the fused embedding of the ith dense-pixel.

However, this loss function is not suitable for symmetric objects which have more than one and possibly an infinite number of canonical frames. Therefore, for symmetric objects, it instead minimizes the distance between each point on the estimated model orientation and the closest point on the ground truth model. The new loss function is defined as :

$$
L_{i}^{p}=\frac{1}{M} \sum_{j} \min _{0<k<M}\left\|\left(R x_{j}+t\right)-\left(\hat{R}_{i} x_{k}+\hat{t}_{i}\right)\right\|
$$

Optimizing over all predicted per dense-pixel poses would be to minimize the mean of the per dense-pixel
losses. Simutaneously, inorder to train the network to learn to balance the confi- dence among the per dense-pixel prediction, the loss function weights the per dense-pixel loss with the dense-pixel confidence, and add a second confidence regularization term:

$$
L=\frac{1}{N} \sum_{i}\left(L_{i}^{p} c_{i}-w \log \left(c_{i}\right)\right)
$$

where N is the number of randomly sampled dense-pixel features from the P elements of the segment and w is a balancing hyperparameter.

### Iterative Refinement 

 A neural network-based iterative refinement module based on densely-fused embedding is proposed that can improve the final pose estimation result in a fast and robust manner without additional rendering techniques.
 The network aims to correct its own pose estimation error in an iterative manner.

The procedure is illustrated in Fig. 3.

<div style="width:50%; margin:0 auto;" align="center" markdown="1">
![formula](/assets/images/DenseFusion_refinement.png " model")
</div>


 The refinement procedure uses the previously predicted pose as an estimate of canonical frame of the target object and transform the input point cloud into this estimated canonical frame.  Then feed the transformed point cloud back into the network and predict a residual pose based on the previously estimated pose. This procedure can be applied iteratively and generate potentially finer pose estimation each iteration.
 
 After K iterations, we obtain the final pose estimation as the concatenation of the per-iteration estimations:

 $$
\hat{p}=\left[R_{K} \mid t_{K}\right] \cdot\left[R_{K-1} \mid t_{K-1}\right] \cdots \cdots\left[R_{0} \mid t_{0}\right]
$$

## Contribution of this paper:

Prior works either extract information from the RGB image and depth separately or use costly post-processing steps, limiting their performances in highly cluttered scenes and real-time applications. 

For example, recent methods that use deep networks for pose estimation from RGB-D inputs, such as PoseCNN and MCN require elaborate post-hoc refinement steps to fully utilize the 3D information, which can't be optimized jointly with the final objective and are prohibitively slow for real-time applications.

Frustrum PointNet  and PointFusion models have achieved good performances in driving scenes and the capacity of real-time inference. However, these methods fall short under heavy occlusion, which is common
in manipulation domains.

The core of DenseFusion is to embed and fuse RGB values and point clouds at a per-pixel level, as opposed to prior work which uses image crops to compute global features  or 2D bounding boxes. This per-pixel fusion scheme can handle heavy occlusion. In addition, an iterative method which performs pose refinement within the end-to-end learning framework. This greatly enhances model performance while keeping inference speed real-time





## Reference: 
C. Wang et al., "DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion," 2019 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), Long Beach, CA, USA, 2019, pp. 3338-3347, doi: 10.1109/CVPR.2019.00346.

[paper](https://arxiv.org/abs/1901.04780) &nbsp;        [github](https://link.zhihu.com/?target=https%3A//github.com/j96w/DenseFusion)