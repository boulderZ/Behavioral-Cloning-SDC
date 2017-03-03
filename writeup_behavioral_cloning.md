#**Behavioral Cloning**

##SDC writeup report for behavioral cloning project

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* mymods.py containing python modules I created that work with model.py
* writeup_behavioral_cloning.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model I used includes three convolutional layers of 3x3 filters of depths 16-32-64. I used 'elu' activation as it was suggested in literature to possibly work better for this type of regression model requiring outputs around zero. I also used dropout layers in the final fully connected layers to help with overfitting. I  (model.py lines 124-144 )

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).
I also experimented with other models including Nvidia and another posted by a former student for comparison and debug (145-192).
####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 137 and 140). I also experimented with adding more dropout layers (and less) as to arrive at my final model.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 97-103). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

In the end, the validation results were not very helpful to determine overfitting. One has to just run the models on the track to determine what performs best.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 144). Used lr = 1 e-4.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving in reverse direction. I also used the udacity supplied training data.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to leverage models similar to Nvidia paper and simplify. Reviewed literature and internet for what models others had used as well. I knew the task was more about the data than the model from reading and forums, so I wanted to spend more time on data and less on model derivation. I used a variation of a simplified model I reviewed on the internet and compared it to Nvidia and another posted by a former student.

At first I used training and validation but immediately noticed that the loss would just track between the two. Therefore looking at validation loss was not very helpful and I abandoned it early on. This was also confirmed in the forums.

I first tried to make it work using only  udacity data. I had limited success with this and decided to collect more data to supplement the udacity data. I collected normal driving, recovery driving and also collected repeated driving through some of the tough  spots like the bridge and dirt road after the bridge.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 124-144) consisted of a convolutional neural network with the following layers and layer sizes:
| Layers        |  Size         |
| ------------- |:-------------:|
| Conv          | 3x3x16        |
| MaxPool       | 2x2           |  
| Conv          | 3x3x32        |
| MaxPool       | 2x2    |
| Conv        | 3x3x64 |
| MaxPool   | 2x2 |
| Fully Connectedect | 500 |
| Dropout      | 0.5 |
| Fully Connect | 100 |
| Dropout  | 0.25 |
| Fully Connect | 20 |
| Fully Connect | 1 |

####3. Creation of the Training Set & Training Process
#### Data collection
I started with the supplied udacity training data and added to it with my own driving on the simulator. I drove around the track normally in both directions, recorded recovery driving, and also recorded extra driving through the bridge and dirt turn off after the bridge. The udacity data had 8036 samples, my own data had 19023 samples for a total of 27,059 original samples before augmentation.

#### Data augmentation
I wanted to augment the data on the fly rather than store augmented images on the hard drive.  My approach was to use only the 27,059 samples and then randomly generate augmented images on the fly for every training epoch. I will discuss how I controlled the distribution of the data in the next section. Here I will detail the augmentation pipeline itself. I used the following types of augmentation and implemented each one randomly: camera angle, flipping, brightness, shadow, shift, cropping. While developing the augmentation I also investigated the speed performance of different methods. When using an inline augmentation pipeline, it is important to consider speed of processing. I will show some of the speed results in an appendix for those interested. One thing interesting that I found: the base random class was faster than the numpy version by a fair amount when used for single values like random.random(). I put comments in my source code wherever I did speed comparisons. Each type of augmentation is described in the table below:

| Augment Type   |  Description        |
| -------------  |:-------------:|
| Camera Angle   | Described in lecture. Use left and right cameras and adjusted steering by +/- 0.25 from original center value|
| Flip           | Described in lecture. Flip the image and use negative of original steer angle |
| Brightness     | Vary brightness of image by multiplying it by a value between 0.3 and 1.3 and then clipping to 255 to avoid the wrap |
| Shadow         | Add simulated shadow to the image. Modified a clever approach from A. Staravoitau |
| Shift          | Shift the image horizontal and adjust steering angle. Modified approach from V.Yadav.
| Cropping       | In addition to normal cropping, vary the position of the crop up and down by small amount. |

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
