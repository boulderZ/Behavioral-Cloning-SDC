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

[figure1]: ./figures/figure_1.png "Augmentation Examples without cropping"
[figure2]: ./figures/pipeline_all.png "Augmentation Examples followed by crop/normalize"
[figure3]: ./figures/combined_steer_before_augment.png "Steering Histogram before binning and augmentation"
[figure4]: ./figures/steer_after_binning.png "Steering Histogram after Binning"
[figure5]: ./figures/combined_steer_augment.png "Sample Steering Histogram after Binning and Augmentation"


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
| ------------- |-------------|
| Input         | 32x128x3
| Conv          | 3x3x16        |
| MaxPool       | 2x2           |  
| Conv          | 3x3x32        |
| MaxPool       | 2x2    |
| Conv        | 3x3x64 |
| MaxPool   | 2x2 |
| Fully Connected | 500 |
| Dropout      | 0.5 |
| Fully Connected | 100 |
| Dropout  | 0.25 |
| Fully Connect | 20 |
| Fully Connect | 1 |

####3. Creation of the Training Set & Training Process
#### Data collection
I started with the supplied udacity training data and added to it with my own driving on the simulator. I drove around the track normally in both directions, recorded recovery driving, and also recorded extra driving through the bridge and dirt turn off after the bridge. The udacity data had 8036 samples, my own data had 19023 samples for a total of 27,059 original samples before augmentation.

#### Data augmentation
I wanted to augment the data on the fly rather than store augmented images on the hard drive.  My approach was to use only the 27,059 samples and then randomly generate augmented images on the fly for every training epoch. I will discuss how I controlled the distribution of the data in the next section. Here I will detail the augmentation pipeline itself. I used the following types of augmentation and implemented each one randomly: camera angle, flipping, brightness, shadow, shift, cropping. While developing the augmentation I also investigated the speed performance of different methods. When using an inline augmentation pipeline, it is important to consider speed of processing. I will show some of the speed results in an appendix for those interested. One thing interesting that I found: the base random class was faster than the numpy version by a fair amount when used for single values like random.random(). I put comments in my source code wherever I did speed comparisons. Each type of augmentation is described in the table below:

| Augment Type   |  Description        |
| -------------  |-------------|
| Camera Angle   | Described in lecture. Use left and right cameras and adjusted steering by +/- 0.25 from original center value|
| Flip           | Described in lecture. Flip the image and use negative of original steer angle |
| Brightness     | Vary brightness of image by multiplying it by a value between 0.3 and 1.3 and then clipping to 255 to avoid the wrap |
| Shadow         | Add simulated shadow to the image. Modified a clever approach from A. Staravoitau |
| Shift          | Shift the image horizontal and adjust steering angle. Modified approach from V.Yadav.
| Cropping       | In addition to normal cropping, vary the position of the crop up and down by small amount. Mod from A. Staravoitau |

Examples of the various augmentation types are shown without cropping in figure 1.

![alt text][figure1]

And with cropping followed by normalization in figure 2.

![alt text][figure2]

The augmentation flow was implemented in `process_image_pipeline()` in lines 123-164 of mymods.py. The order was: camera angle, brightness,shadow,shift,flip, cropping. Camera angle was experimented with but not used in the model turned in for this project. Brightness,shadow,and shift were all applied with a random probability of 0.3. Flipping was applied with probability of 0.5. The individual functions for each augmentation type are all in mymods.py.

#### Data distribution

The distribution of angles is very important for the model to work. There are too many samples near zero and there is not an even distribution of angles (classes) from either the Udacity data set or data taken from the training simulator. This can be observed by looking at histograms of the angles. The raw histogram before any manipulation of the data is shown in figure 3.

![alt text][figure3]

At each epoch beginning, the original data samples (27,059) are shuffled and then passed to a binning algorithm  (`distribute_samples()`, lines 357-389 in mymods.py) that returns a binned version of the original data with a maximum count for each angle bin. This helps to distribute the classes (angles) so that there will be less of a bias towards going  straight. Since the binned samples are drawn from a shuffled version of the original samples, there will be variety in any classes that had more than the max count of angles in the original data.  A sample histogram of this binned result is shown in figure 4.

![alt text][figure4]

The binned samples are then put into batches for processing in the generator function. The loop for processing each image has further data distribution  controls to limit the samples near zero angles before augmentation. The loop will keep angles within a range with a probability and range that can be programmed. I used a probability of 0.3 and a range of -0.2 to 0.2. That means it keeps 30% of all values within the range of -0.2 to 0.2. Any angles outside of that range are not limited.

The augmentation will change the distribution of angles when any of the augment types of Camera Angle, Shift, or Flip are used. I only used shift and flip in the model turned in for this assignment. An example of the angle histogram after augmentation is shown in figure 5. This was produced using ( `test_distribution()` , lines 245-282 ) in mymods.py.

![alt text][figure5]


### Conclusions and further work
The car drives around track one but does not generalize to track 2. To do this would require a lot more training data and further augmentation. Also the model would benefit from further control of the data distribution. This version of my code does not completely control the distribution of the angles and I believe suffers in performance because of that. I have been working on another version which I will post separately but for now I need to move on to the other projects. I have completed code that will downsample the original data in a way that will account for stopping and starting the recording in different portions of the track. This is to reduce the number of essentially duplicate images that result from the FPS (frames per second) being so high. The Nvidia paper mentions this as well. The other thing I am doing is to simply save the augmented images in a directory so that I can have more control over the final distribution when binning the data.
