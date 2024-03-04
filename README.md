# Retail-Inventory-Optimization-by-Shelf-Monitoring-and-Product-Detection

At a typical retail store employees manually track and verify what products are available in the store. Employees count each product and record information about the item, such as pricing, on a platform. Manually counting the products and filling out the information is a slow process, and employees are prone to making errors.
The success of deep learning to solve complex problems is not hidden from anyone these days. Deep learning is helping automate problems in all walks of life. For this project, I present the application of deep learning-based algorithm, to detect empty inventory in retail stores.
Normally, when a customer visits a retail store, and sees a shelf that doesn’t have the product they need, then many customers might leave the store without asking the employees if they have that item. Even if the store had that item in their warehouse. This can cause the store to lose out on potential sales for as long as the display inventory remains empty. The empty shelf detection system I designed could help stores restock inventory quickly thereby reducing loss of sales and maintaining customer footfall.

![intro][0]

## The project detects empty shelf from a video stream and identifies the products to restock

![resnet_result][11]
![output][5]

## Overview

- Empty shelves cost U.S. retailers alone $82 billion in missed sales as of 2021. 
- In fact, on average, “out of shelf” items cost retailers $1.4 billion every week. 
- A 2021 survey showed that when confronted with empty shelves, 
  - 20% of consumers postponed their purchase
  - 10% of consumers purchased the item elsewhere
  - 16% of consumers shifted to an online source
- Leading to retailers losing 46% of possible sales. 
- The goal of this project is to develop a robust solution to promptly identify out of stock products and restock empty shelves using deep learning models 

## Dataset

The dataset contains a total of 20000 images which were collected by following means:

- 75% From SKU110K dataset [here](https://universe.roboflow.com/sku-fzp1x/sku110k-toc01)
- 20% Manually clicked from different retail stores
- 5% Scraped from the internet
- All the images were annotated / labeled using label img
- 80/20 train-test split was done for all the models
- Different types of images are used to train models such as 
  - Cropped
  -	Padded 
  -	Different angles 
- Various departments such as frozen food, cold drinks refrigerator are also used to train the models

![data][1]

## Project Architecture

![architecture][2]

#### Input

The input for the inference could be in the form of image or video in any format and pixel dimensions. The video could be a live stream from any video capturing device or in the form of saved video.

#### Model

The model used for inference is Faster R- CNN with Resnet 101 as it had the minimum loss amongst the other two models, which are ssd with inception v2 and yolo v5. The inference time by resnet is much more than that of ssd and yolo but the accuracy of resnet is far superior.

#### Prediction

The prediction from the model is in the form of bounding box which is used to calculate the center co-ordinate of the bounding box.

![prediction][3]

#### Database

Pymongo is used as the database, it has a list of products present on any given retail store isle along with its bounding box co-ordinates.

![database][4]

#### Output

The center co-ordinates of the bounding box from the model is used to extract the product name saved in the database by referring to the co-ordinates of the product in the database.

![output][5]

## Models Used and Performance Comparison

#### Models used in this project are:
 - SSD Inception V2
 - YOLO V5
 - Resnet 101 Faster RCNN (Best Performing and model used in this project)

#### SSD Inception V2

The most effective and accurate deep convolutional neural network (faster region-based convolutional neural network (Faster R-CNN) Inception V2 model, single shot detector (SSD) Inception V2 model) based architectures for real-time hand gesture recognition is proposed. The proposed models are tested on standard data sets (NUS hand posture data set-II, Senz-3D) and custom-developed (MITI hand data set (MITI-HD)) data set. The performance metrics are analysed for intersection over union (IoU) ranges between 0.5 and 0.95. IoU value of 0.5 resulted in higher precision compared to other IoU values considered (0.5:0.95, 0.75). It is observed that the Faster R-CNN Inception V2 model resulted in higher precision (0.990 for APall, IoU = 0.5) compared to SSD Inception V2 model (0.984 for all) for MITI-HD 160. The computation time of Faster R-CNN Inception V2 is higher than the SSD Inception V2 model and resulted in fewer mispredictions. Increasing the size of samples (MITI-HD 300) resulted in an improvement of APall = 0.991. Improvement in large (APlarge) and medium (medium) size detections are not significant when compared to small (small) detections. It is concluded that the Faster R-CNN Inception V2 model is highly suitable for real-time hand gesture recognition.

#### SSD Loss Graph is displayed below

![ssd_loss][6]

#### SSD Model Result on the image is displayed below

![ssd_result][7]

#### YOLO V5

YOLO is an abbreviation for the term ‘You Only Look Once’. This is an algorithm that detects and recognizes various objects in a picture (in real-time). Object detection in YOLO is done as a regression problem and provides the class probabilities of the detected images. It employs convolutional neural networks (CNN) to detect objects in real-time. As the name suggests, the algorithm requires only a single forward propagation through a neural network to detect objects. This means that prediction in the entire image is done in a single algorithm run. The CNN is used to predict various class probabilities and bounding boxes simultaneously.


#### YOLO Loss Graph is displayed below

![yolo_loss][8]

#### YOLO Model Result on the image is displayed below

![yolo_result][9]

#### Resnet 101

It is a state-of-the-art object detection networks which depends on region proposal algorithms to hypothesize object locations. Advances like SPPnet and Fast R-CNN have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. This model uses a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and object scores at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. RPN and Fast R-CNN into a single network by sharing their convolutional features---using the 'attention' mechanisms, the RPN component tells the unified network where to look. For the very deep VGG-16 model, the detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy.


#### Resnet 101 Loss Graph is displayed below

![resnetloss][10]

#### Resnet 101 Model Result on the image is displayed below

![resnet_result][11]




[0]: images/intro.png
[1]: images/data.png
[2]: images/architecture.png
[3]: images/prediction.png
[4]: images/database.png
[5]: images/output.png
[6]: images/ssd_loss.png
[7]: videos/ssd_result.mp4
[8]: images/yolo_loss.png
[9]: videos/yolo_result.mp4
[10]: images/resnet_loss.png
[11]: videos/resnet_result.mp4