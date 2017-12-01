# **Behavioral Cloning** 

## Writeup, Joseph Rogers

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/figure1.png "Loss Graph"
[image2]: ./images/forward_center.jpg "Center Driving"
[image3]: ./images/reverse_driving.jpg "Reverse Driving"
[image4]: ./images/left_camera.jpg "Left Camera"
[image5]: ./images/near_edge.jpg "Near Edge"
[image6]: ./images/near_dirt.jpg "Near Dirt"
[image7]: ./images/orig.jpg "Original Image"
[image8]: ./images/processed.jpg "Processed Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

Here is a link to my full [github repository.](https://github.com/josephRog/CarND-Behavioral-Cloning-P3)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a slighly modified verion of the NVIDIA deep neural network for training self-driving cars. The original paper describing the network can be found [here.](https://arxiv.org/pdf/1704.07911.pdf)

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 320x160x3 RGB image                             | 
| Normalization       | Input = 320x160x3, Output = 320x160x1   |
| Cropping2D    | Input = 320x160x1, Output = 320x90x1       |
| Convolution 24,5,5   | 2x2 stride, RELU Activation|
| Convolution 36,5,5    | 2x2 stride, RELU Activation|
| Convolution 48,5,5    | 2x2 stride, RELU Activation|
| Convolution 64,3,3    | 1x1 stride, RELU Activation |
| Convolution 64,3,3    | 1x1 stride, RELU Activation |
| Dropout Layer | keep_prob = 0.5           |   
|Flatten   |
| Fully Connected | Dense(100) |
| Fully Connected | Dense(50) |
| Fully Connected | Dense(10) |
| Fully Connected | Dense(1) |

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

This network is slighlty different from the NVIDIA version due to the inclution of a dropout layer before the flatten step. I also removed one of the fully-connected layers.

When training the model I used a generator to save memory with a batch size of 32. The generator function automatically shuffled the data every time it was called. I only even trained over the course of 5 epochs due to diminishing returns of training and validation accuracy. To be mindful of not overfitting the data, I graphed the loss of the two data sets to make sure then were in line with each other.

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

The model has a dropout layer just before the flatten before all of the fully connected layers. The dropout probability was set to 0.5.

The training data used a 80/20 split between the main set and validation data. Training data was augmented by creating mirrored copies of all of the images. The data was mirrored to help reduce left steering bias.

I also used the left and right side cameras on top of the car to give additional data to the network. Each of these cameras was given an extra 0.5 degrees of correction to their steering measurements.

The final model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Creation of the Training Set & Training Process

Ir order to capture good training data, I started with 3 laps of controlled center lane driving around the forward direction of the track. In order to create some variation, I then drove the track in the opposite direction, also in a controlled manner.

Center Lane Driving
![alt text][image2]

Reverse Direction Driving
![alt text][image3]

As mentioned before, I used the left and right side cameras on top of the car create additional data that I hoped would help keep the car centered. The steering correction factor was initially set to 0.1 degrees, but I noticed vasty improved results when setting it a bit higher to 0.5 degrees.

Left Camera
![alt text][image4]

Things were now going generally okay but to help the vehicle recover better, I drove the whole track again but in the shape of a sin curve between each of the edges of the road. I only recorded data during the times when I was driving towards the center of the road, never towards the edges.

Near Road Edge
![alt text][image5]

At this point when testing the model, the entire track could be driven consistently aside from a few key area that were occationally causing trouble.

When the car was about to drive onto the brige or past the dirt on the far side of the bridge, it would sometimes get confused and fail to steer correctly. I fixed this by recording some extra data of how to recover in these key areas. Once this data was added to the training set, the car drove by both areas without a problem.

Driving Near Dirt
![alt text][image6]

Once all the data was gathered I had just under 7000 images of 320x160x3 size. To improve performance and reduce training time, I converted the images to grayscale and cut 70 pixels from the vertical dimension. This left the images with a size of 320x90x1. This new images use about 5 times less memory than the originals.

Original Image
![alt text][image7]

Processed Image
![alt text][image8]

When all said and done, the entire model was able to train in about 1 minute on an NVIDIA GeForce 980TI graphics card.

#### 5. Extra Thoughts

The quality of the training data seemed to make an enourmous differnce on how well the model trained. It was really a case of garbage in garbage out. I had to scrap my data several times and start from scratch.

Using the mouse to provide smoothing analog input for the steering as opposed to the keyboard made a big difference.

Special thanks to Paul Heraty for his quick [guide](https://slack-files.com/T2HQV035L-F50B85JSX-7d8737aeeb) on tips and tricks for how to get started with this project.