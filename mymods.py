##### modules for behavioral  learning project
import numpy as np
from scipy.misc import imresize
from cv2 import imread
from cv2 import warpAffine
import random
import csv
import matplotlib.pyplot as plt


def crop_resize(image, center = 1,augment=0,crop_top = .375, crop_bottom = .125,newsize=(32,128,3)):
    """
    Imports:
        import numpy as np
        from scipy.misc import imresize
    Description:
        Crop and resize image.
        Modified from post by Alex Staravoitau
    Inputs:
        image = single image of shape (row,col, channels)
        center = if 1, subtract 0.5 after reducing range to [0,1]
        crop_top = fraction of top to remove
        crop_bottom = fraction of bottom to remove
        newsize = (h,w,channels)
    Returns:
        image cropped, resized and scaled to -0.5 to 0.5 if center =1

    """
    if augment:
        crop_delta = np.random.uniform(-.05,.05)  # np faster in this categorical_crossentropy
    else: crop_delta = 0
    crop_top = crop_top + crop_delta
    crop_bottom = crop_bottom + crop_delta
    top = int(crop_top * image.shape[0])
    bottom = int(crop_bottom * image.shape[0])
    image = imresize(image[top:-bottom, :], newsize)
    if center:
        image = (image/255.) - 0.5
    else:
        image = image/255.
    image = image.astype(np.float64)
    return image

def shift_image_horiz(image,angle,pixel_to_angle=.004,pixel_shift=50.):
    '''
    Imports:
        import numpy as np
        from cv2 import warpAffine
        import random
    Description:
        Shift image left or right randomly and calculate new steering angle
        use cv2.warpAffine with transfomation matrix M
        Modified from post by Vivek Yadav
    Inputs:
        image = cv2.imread image of shape (160,320,3)
        If image shape is different, likely need to re-calibrate pixel_to_angle
    '''
    rows,cols = image.shape[:2]
    tx = np.random.uniform(-pixel_shift,pixel_shift) # np faster in this case
    new_angle = angle + tx * pixel_to_angle
    ty = 0
    M = np.float32([[1,0,tx],[0,1,ty]])
    return warpAffine(image,M,(cols,rows)),new_angle


def image_brightness(image):
    '''
    Imports:
        import numpy as np
    Description:
        Simulate night and daytime by changing image brightness randomly.
        Saturate pixels to 255 with np.clip if input is uint8
    Inputs:
        image (numpy array cv2.imread style)shape = (h,w,channels)
    Returns:
        image of same type as input with random brightness
    '''
    image_dtype = image.dtype
    alt_val = np.random.uniform(.3,1.3)  # np is faster in this case
    return np.clip((image * alt_val),0,255).astype(image_dtype)


def image_shadow(image,alt_val = 0.5):
    '''
    Imports:
        import numpy as np
        import random
    Description:
        Simulate shadow by reducing all pixel values in a slice by half.
        Slice is across the color channels so all r,g,and b values are
        reduced by half.
        Each slice is shape (1,c,3) where c is the point on the line defined
        by (xtop,0) and (xbot,h-1). Image.shape = (h,w,channels)
        The y axis is zero at the top of the image.
        Modified from post by Alex Staravoitau
    Inputs:
        image = (numpy array cv2.imread style)
        alt_val = fraction to multiply pixel values by (default = 0.5)
    Returns:
        image of same type as input with shadow.
    NOTE: this function changes the original image in the outer scope
    '''
    image_dtype = image.dtype
    h, w = image.shape[0], image.shape[1]
    #[xtop, xbot] = np.random.choice(w, 2, replace=False)
    xtop = random.randint(0,w-1)  # 10x faster than np.random.choice
    xbot = random.randint(0,w-1)
    m = h / (0.5 + (xbot - xtop))  # approx slope, take care of div by zero
    b = - m * xtop
    for i in range(h):
        c = int((i - b) / m)
        image[i, :c, :] = image[i, :c, :] * alt_val
    return image.astype(image_dtype)

def augment_brightness(image):  ## from Vivek Yadov for speed compare only
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()  # np faster in this case
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def process_image_pipeline(line_sample,augment_prob,crop_on=1,newsize=(32,128,3),
                             mult_camera=True):
    '''
    Imports:
        import random
        from cv2 import imread
        others from other functions
    Description:
        Image processing pipeline for SDC simulator. Process one line
        of data csv file at time, augment data if augment=1.
        Data is augmented randomly so every type of augmentation is not
        always performed.
    Inputs:
        line_sample = a row from the data csv log file
        augment_prob = probility of including each augment type, if zero None
        crop_on = turn on cropping
        newsize = size to crop image to
        mult_camera = turn multiple cameras if True
    Returns:
        image = processed and or augmented image
        steer = steering angle value
    '''
    steer_correct = [0.0, 0.25, -0.25]  # [center,left,right]
    if  mult_camera:
        rand_index = random.randint(0,2)  # random in [0,1,2], faster than np
    else:
        rand_index = 0
    image = imread(line_sample[rand_index]) # get image
    steer = float(line_sample[3]) + steer_correct[rand_index]
    if random.random() < augment_prob:  # faster than np version
        image = image_brightness(image)
    if random.random() < augment_prob:
        image = image_shadow(image)
    if random.random() < augment_prob:
        image,steer = shift_image_horiz(image,steer)
    if random.random() < 0.5:       # flip images with probability of 1/2
        image = np.fliplr(image)
        steer = - steer
    if crop_on:
        image = crop_resize(image,augment=augment_prob,newsize=newsize)

    return image,steer, rand_index


def combine_images(img_array,xdim,ydim,transpose = 0):
    '''
    Imports:
        import numpy as np
    Description:
        Combines multiple images into a single image for plotting.
        Plots single dimension input array with column incrementing fastest
    Inputs:
        img_array =  1D array of images (cv2.imread format) of shape (h,w,ch)
        xdim = number of rows of images
        ydim = number of cols of images
        tranpose = if 1, then reverse order of xdim,ydim and transpose img_array
    Outputs:
        one_plot = combined single image of shape (h*xdim,w*ydim,ch)
    '''
    if xdim * ydim > len(img_array):
        print('ERROR: combine_images(): xdim * ydim > len(img_array)')
        return 0
    if transpose:  # flip xdim and ydim
        new_array=[]
        for i in range(ydim):
            for j in range(xdim):
                new_array.append(img_array[i+j*ydim])
        img_array = new_array
        tmp = ydim
        ydim = xdim
        xdim = tmp
    h,w,channels = img_array[0].shape
    one_plot = np.zeros((xdim*h, ydim*w, channels))
    indx = 0
    for row in range(xdim):
        for col in range(ydim):
            one_plot[row*h:(row*h+h), col*w:(col*w+w), :] = img_array[indx]
            indx += 1
    return one_plot.astype(np.uint8)

def show_image_pipeline(num_origs,num_gen,crop_on=0):
    '''
    Imports:
        import random
        from cv2 import imread
        import csv
    Description:
        Create array of images showing the original image followed by the
        augmented images before cropping and normalization.
    Inputs:
        num_origs = number of original images to augment
        nun_gen = number of generated images for each original
    Returns:
        img_array = array of images with order (orig1,mods...,orig2,mods...)
        steer_array = array of steering angles for each image in img_array
    '''
    samples = []
    with open('./data/data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    samples = samples[1:]    ##  Remove header
    img_array=[]
    steer_array = []
    augment_prob = 0.33
    for i in range(num_origs):
        rand_indx = random.randint(0,len(samples)-1)
        line_sample = samples[rand_indx]

        for j in range(num_gen):
            image,steer,im_indx = process_image_pipeline(line_sample,
                                                augment_prob,crop_on=crop_on)
            if j == 0:
                name = './data/data/IMG/'+line_sample[im_indx].split('/')[-1]
                orig_image = imread(name)
                img_array.append(orig_image)   # original image
                steer_array.append(float(line_sample[3]))  # original steer

            img_array.append(image)    # append augmented image
            steer_array.append(steer)
    return img_array,steer_array

def test_distribution(zero_thresh=1.0,augment_prob=0.75,zero_reject_range = .05,newsize=(32,128,3)):
    ############## read in data supplied by Udacity #############
    '''
    batch_size = 32
    zero_thresh = 1.0
    augment_prob = 0.75
    zero_reject_range = 0.05
    '''
    batch_size = 32
    crop_on = 1
    samples = load_samples()
    np.random.shuffle(samples)
    bin_samples = distribute_samples(samples)  # get reduced count samples
    num_samples = len(bin_samples)
    angles = []
    steer_orig = []
    for offset in range(0, num_samples, batch_size):
        batch_samples = bin_samples[offset:offset+batch_size]
        for batch in batch_samples:
            if zero_thresh < 1.0:
                keep_search = 1
                while keep_search:
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
                                                    newsize)
            angles.append(angle)
            steer_orig.append(float(line_sample[3]))
    return angles,steer_orig

def test_distribution_old(size=10000):
    ############## read in data supplied by Udacity #############
    samples = load_samples()
    np.random.shuffle(samples)
    steer_array = []
    steer_orig = []
    augment_prob = 0.33
    zero_reject = 1     # if set reduce samples with zero angle
    # augment_prob = 0.0
    # zero_reject = 0
    crop_on = 1
    for i in range(size):
        if zero_reject:
            keep_search = 1
            while keep_search:
                rand_indx = random.randint(0,len(samples)-1)
                line_sample = samples[rand_indx]
                if float(line_sample[3]) == 0.0:
                    if random.random() < 0.03:
                        keep_search = 0
                else:
                    keep_search = 0
        else:
            rand_indx = random.randint(0,len(samples)-1)
            line_sample = samples[rand_indx]
        image,steer,im_indx = process_image_pipeline(line_sample,
                                                augment_prob,crop_on)
        steer_array.append(steer)
        steer_orig.append(float(line_sample[3]))
    return steer_array,steer_orig

### plot a histogram from numpy array
def plot_histogram(x):
    '''
    Imports:
        from numpy import np
        import matplotlib.pyplot as plt
    Description:
        From stack exchange:
        http://stackoverflow.com/questions/30112420/histogram-for-discrete-values-with-matplotlib
        This one will always capture correct counts, plots accurate histogram
    '''
    data = np.asarray(x)
    d = np.diff(np.unique(data)).min()
    if d < 0.001:
        d = 0.001     # plt.hist() takes forever is d is much smaller
    left_of_first_bin = data.min() - float(d)/2
    right_of_last_bin = data.max() + float(d)/2
    out = plt.hist(data, np.arange(left_of_first_bin, right_of_last_bin + d, d))
    plt.show()

def load_samples(use_mydata = 1):
    samples = []
    with open('./data/data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    samples = samples[1:]    ##  Remove header
    for i in range(len(samples)):
        for j in range(3):
            samples[i][j] = './data/data/IMG/' + samples[i][j].split('/')[-1]  # add path
    ###### add my training data ###########################
    if use_mydata:
        with open('driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)
        with open('/home/ai/CarND-Cloning/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)
    return samples

def distribute_samples(samples,max_count=200):
    '''
    Imports:
        from numpy import np
    Description:
        Find all counts of each angle bin and only keep up to 200 samples from
        each bin.
    Inputs:
        samples = list of lines from csv file from simulator
        max_count = maximum count to keep
    Returns:
        new_samples = new numpy array with fewer items
    '''
    ####  Get angle data
    angle_array = []
    for samp in samples:
        angle_array.append(float(samp[3]))
    ### get counts and locations
    nsamples = np.array(samples)
    unq,counts = np.unique(angle_array,return_counts = True)
    angle_indx = np.split(np.argsort(angle_array), np.cumsum(counts[:-1]))
    ### get indices to keep
    keep_list = []
    for item in angle_indx:
        if len(item) > max_count:
            keep_list.append(item[:max_count])
        else:
            keep_list.append(item)
    keep_list = np.asarray(keep_list)
    ### flatten the list (only np.hstack works here, not np.flatten)
    keep_list = np.hstack(keep_list)
    ### return new list
    return nsamples[keep_list]

def get_angles_from_samples(samples):
    angle_array = []
    for samp in samples:
        angle_array.append(float(samp[3]))
    return angle_array

def plot_tmp():
    '''
    quick plot to avoid typing in terminal
    '''
    fig = plt.figure(figsize = (10,10))
    fig.subplots_adjust(hspace=0,wspace=0)
    sub_i = 1
    for i in range(0,len(img_array),4):
        axis =fig.add_subplot(5,5, sub_i , xticks=[], yticks=[])
        axis.imshow(img_array[i])
        sub_i +=1
    for i in range(1,len(img_array),4):
        axis =fig.add_subplot(5,5, sub_i, xticks=[], yticks=[])
        axis.imshow(img_array[i])
        sub_i+=1
    for i in range(2,len(img_array),4):
        axis =fig.add_subplot(5,5, sub_i, xticks=[], yticks=[])
        axis.imshow(img_array[i])
        sub_i+=1
    for i in range(3,len(img_array),4):
        axis =fig.add_subplot(5,5, sub_i, xticks=[], yticks=[])
        axis.imshow(img_array[i])
        sub_i+=1
    for i in range(3,len(img_array),4):
        axis =fig.add_subplot(5,5, sub_i, xticks=[], yticks=[])
        axis.imshow(img_array[i]-0.5)
        sub_i+=1
    return
