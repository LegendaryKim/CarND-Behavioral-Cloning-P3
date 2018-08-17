# Udacity - Self-Driving Car NanoDegree: Behaviorial Cloning Project

[//]: # (Image References)

[NVIDIA_model]: ./examples/cnn-architecture-624x890.png "NIVIDIA Model"
[Mymodel]: ./examples/Model.png "Model Visualization"
[Center]: ./examples/center.png "Center Image"
[Crop]: ./examples/center_cropped.png "Cropped Image"
[Resize]: ./examples/center_cropped_resized.png "Resized Image"
[flip]: ./examples/center_flipped.png "Flipped Image"
[Center_Left_RIght]: ./examples/Center_Left_Right.png "Center, Left, Right Images"
[MSE]: ./examples/figure_new_30_32_05_025_001_resize_BETTER.png "MSE"
[Reuslt_Gif]: ./run_final_gif.gif "Result_Gif"

My Result:

![alt text][Reuslt_Gif]

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

---
### Files Submitted & Code Quality

My project includes the following files:
* clone_gen.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_gen.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* run_final.mp4 recording of your vehicle driving autonomously around the track

#### Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### My model architecture

My first try out is LeNet, but, I decided to utilize the NVIDIA architecture as a training model:
![alt text][NVIDIA_model]

From NIVIDA model, the cropping, resizing and normalizing layers are applied on input data. Its model includes dropout layers after CNN and dense layers. Also, the model utilizes ReLu activation functions to introduce nonlinearity.

The final model summay is as follow:

|Layer  |
|-----------|
|Cropping2D |
|Lambda_resizing|
|Lambda_normalizing|
|Convolution2d_relu|
|Dropout|
|Convolution2d_relu|
|Dropout|
|Convolution2d_relu|
|Dropout|
|Convolution2d_relu|
|Dropout|
|Convolution2d_relu|
|Dropout|
|flatten|
|Dense|
|Dropout|
|Dense|
|Dropout|
|Dense|
|Dropout|
|Dense|

#### Reduce overfitting with dropout and tunning parmeters with validation sets

The model contains dropout layers with 50% dropout-rate in order to reduce overfitting.

Furthermore, the input data is divided to training (80%) and validation (20%) sets to ensure that the model was not overfitting. Lastly, the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### Appropriate training data

Training data provided by Udacity was chosen to keep the vehicle driving on the road as three images: center, left and right cameras. The left and right images are utilized with correction angles $\pm 0.25^{\circ}$. Also, to control the biased steering_correction, I dropped out 50% of small steering angle data. All data was flipped to make the model follow clickwise curves properly (the training data is mainly counter-clickwise curved)

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet model with 5 epochs and the training data provided by Udaciy. The MSE of training set went down at every epochs, but the one of validation was stayed higher value (it means overfitting). Moreover, in the test, the vehicle didn't follow the curved track and rushed into the lake. 

To overcome these mentioned problems, I applyed pre-processing steps on the image and steering angle data. For example, I cropped the images from 160x320 to 75x320, resized them to 64x64, and normalized the data. Also, I flipped the images horizontally and train the model with the flipped with their negative steering angles.

The next step was to use a more powerful model, NIVIDA architecture. The dropout layers were applied in the model to control the overfitting. I utilized the Amazon AWS to train the model with 30 epoches.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### Final Model Architecture

Visualization of the final model architecture is given as: 

![alt text][Mymodel]

#### Creation of the Training Set & Training Process

To capture good driving behavior, I used the data from Udacity. Here is an example image of center lane driving:

![alt text][Center]

The training set included a lot of small steering angles, and I dropped the data with smaller steering angles than $ 0.01 ^{\circ}$ randomly with 50% probability.

To argument the data set, I also flipped images and steering angles:
![alt text][flip]

Furthermore, the images of left and right cameras also were included with corrected steering angles ($ \pm 25 ^{\circ}$):
![alt text][Center_Left_Right]

After the collection process, I had 27872 number of data points. I then preprocessed this data by cropping and resizing them to 64 by 64.

![alt text][Crop]
![alt text][Resize]

The ideal number of epochs was 30 as evidenced, the time history of training and validation losses is followed by:

![alt text][MSE]

 I used an adam optimizer so that manually training the learning rate wasn't necessary.
