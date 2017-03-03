# TODO: Build a model
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D,MaxPooling2D
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
tf.python.control_flow_ops = tf    # not sure if this is needed still
import random
#import os
import csv
from mymods import *
from cv2 import imread
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as skshuffle

############## Define Generator #############

def generator(samples,newsize,batch_size=32,max_count=200):
    zero_thresh = 0.3
    augment_prob = .3
    zero_reject_range = 0.2
    crop_on = 1
    mult_camera = False
    increment_zero_thresh = 0
    loop_count = 0
    while 1: # Loop forever so the generator never terminates
        print('  GETTING NEW SAMPLES ' )
        print('loop_count = ', loop_count)
        np.random.shuffle(samples)
        # create binned version of larger sample base with flatter distribution
        bin_samples = distribute_samples(samples,max_count=max_count)
        num_samples = len(bin_samples)
        loop_count += 1
        if loop_count > 2 and increment_zero_thresh:
            zero_thresh += 0.1  # experimented with this, but not used in end
            print('zero_thresh = ',zero_thresh)
        for offset in range(0, num_samples, batch_size):
            batch_samples = bin_samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch in batch_samples:
                if zero_thresh < 1.0: # if rejecting small angles
                    keep_search = 1
                    while keep_search: # search for sample meeting criteria
                        rand_indx = random.randint(0,len(bin_samples)-1)
                        line_sample = bin_samples[rand_indx]
                        if abs(float(line_sample[3])) < zero_reject_range:
                            if random.random() < zero_thresh:
                                keep_search = 0
                        else:
                            keep_search = 0
                else:
                    line_sample = batch

                image,angle,im_indx = process_image_pipeline(line_sample,
                                                        augment_prob,crop_on,
                                                        newsize,
                                                        mult_camera=mult_camera)
                images.append(image)
                angles.append(angle)


            X_train = np.array(images)
            y_train = np.array(angles)
            yield skshuffle(X_train, y_train)


# Main code here
# load data samples from udacity and my sim data
samples = load_samples(use_mydata = 1) ## use_mdata = 0, load udacity data only
np.random.shuffle(samples)
# choose which model to use
use_model = 'mine'
#use_model = 'vivek'
#use_model = 'nvidia'
compile_model = 1 # compile model or use previous one in directory
validate = 0  # split into training and validation or just run training only
batch_size = 32
num_epochs = 30
max_count = 200  # maximum count returned from distribute_samples()

if use_model == 'mine':
    newsize = (32,128,3)
elif use_model == 'vivek':
    newsize = (64,64,3)
elif use_model == 'nvidia':
    newsize = (66, 200, 3)
else:
    print('no model specied')

### samples_per_epoch needs to match what is used in generator
##  create generators for training and validation 
if validate:
    train_samples, validation_samples = train_test_split(samples, test_size=0.2,
                                                          random_state=42)
    num_train_bin_samples = len(distribute_samples(train_samples))
    num_validation_bin_samples = len(distribute_samples(validation_samples))
    train_generator = generator(train_samples,newsize,batch_size=batch_size)
    validation_generator = generator(validation_samples,newsize,batch_size=batch_size)
else:
    train_generator = generator(samples,newsize,batch_size=batch_size)
    num_train_bin_samples = len(distribute_samples(samples,max_count=max_count)) # use for samples_per_epoch

############################  TEST GENERATOR #################
# import matplotlib.pyplot as plt
# testg = next(train_generator)    ### get next iteration
# print('testg Y = ',testg[1])
# print('testg dtype = ',testg[1].dtype)
# img = testg[0][5]
# print('max img/min img = ',np.max(img),np.min(img))
# print('img dtype = ',img.dtype)
#
# plt.figure()
# plt.imshow(img)
# plt.show()
###################################################################

####################### TEST WITH OTHER MODELS ####################

if compile_model==1 and use_model == 'mine':
    print('compiling and using use_model mine = ', use_model)
    input_shape = (32,128,3)
    model = Sequential()
    #model.add(Convolution2D(3,1,1,input_shape=input_shape,activation='elu'))
    model.add(Convolution2D(16, 3, 3, input_shape=input_shape, activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, 3, 3, activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Dropout(.5))
    model.add(Activation('elu'))
    model.add(Dense(100))
    model.add(Dropout(.25))
    model.add(Activation('elu'))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=1e-04), loss='mean_squared_error')
    #model.compile(optimizer='adam',loss='mse')

elif compile_model==1 and use_model == 'vivek':
    print('compiling and using use_model vivek = ', use_model)
    input_shape = (64,64,3)
    model = Sequential()
    model.add(Convolution2D(3,1,1,input_shape=input_shape,activation='elu'))
    model.add(Convolution2D(32, 3, 3,  activation='elu'))
    model.add(Convolution2D(32, 3, 3,  activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.5))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.5))
    model.add(Convolution2D(128, 3, 3, activation='elu'))
    model.add(Convolution2D(128, 3, 3, activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('elu'))
    model.add(Dense(64))
    model.add(Activation('elu'))
    model.add(Dense(16, activation='elu'))
    model.add(Dense(1))
    #model.compile(optimizer=Adam(lr=1e-04), loss='mean_squared_error')
    model.compile(optimizer='adam',loss='mse')

elif compile_model == 1 and use_model == 'nvidia':
    print('compiling and using use_model nvidia = ', use_model)
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, input_shape=(66, 200, 3), activation='elu',
                              subsample = (2,2),border_mode='valid',init = 'he_normal'))
    model.add(Convolution2D(36, 5, 5,  activation='elu',
                              subsample = (2,2),border_mode='valid',init = 'he_normal'))
    model.add(Convolution2D(48, 5, 5,  activation='elu',
                              subsample = (2,2),border_mode='valid',init = 'he_normal'))
    model.add(Convolution2D(64, 3, 3,  activation='elu',
                              subsample = (1,1),border_mode='valid',init = 'he_normal'))
    model.add(Convolution2D(64, 3, 3,  activation='elu',
                              subsample = (1,1),border_mode='valid',init = 'he_normal'))
    model.add(Flatten())
    model.add(Dense(1164,init='he_normal',activation='elu'))
    model.add(Dense(100,init='he_normal',activation='elu'))
    model.add(Dense(50,init='he_normal',activation='elu'))
    model.add(Dense(10,init='he_normal',activation='elu'))
    model.add(Dense(1,init='he_normal'))
    model.compile(optimizer=Adam(lr=1e-04), loss='mean_squared_error')
else:
    model = load_model('model.h5')
    print('no compile, reloading model = ',use_model)


save_epoch_model = ModelCheckpoint(filepath='./modelruns/model.{epoch:02d}.hdf5',
          verbose=0, save_best_only=False, save_weights_only=False,
          mode='auto', period=1)
if validate:
    history=model.fit_generator(train_generator,
                                 samples_per_epoch=num_train_bin_samples,
                                  validation_data=validation_generator,
                                  nb_val_samples=num_validation_bin_samples,
                                  nb_epoch=num_epochs,
                                  callbacks = [save_epoch_model])
else:
    history=model.fit_generator(train_generator,
                                 samples_per_epoch=num_train_bin_samples,
                                 nb_epoch=num_epochs,
                                 callbacks = [save_epoch_model])

model.save('model.h5')
########################### Test prediction ##################
batch_sample = samples[50]
name = batch_sample[0]
center_image = imread(name)
center_angle = float(batch_sample[3])
#center_angle = float(batch_sample[3]) + steer_correct[rand_index]
center_image = crop_resize(center_image,center=1,newsize=model.input_shape[1:]) # resize/crop
steering_angle = float(model.predict(center_image[None, :, :, :], batch_size=1))
print('steering angle = ',steering_angle, 'actual angle = ',center_angle)
print('inline = ',model.input_shape[1:])
#print('model.nb_epoch = ',model.nb_epoch)  does not work, not
