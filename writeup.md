**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following: \* Load the data set (see
below for links to the project data set) \* Explore, summarize and visualize the
data set \* Design, train and test a model architecture \* Use the model to make
predictions on new images \* Analyze the softmax probabilities of the new images
\* Summarize the results with a written report

Rubric Points
-------------

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project
code](https://github.com/vmjfk/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs
data set:

-   Number of training examples = 34799

-   Number of testing examples = 4410

-   Number of validation examples = 12630

-   Image data shape = (32, 32, 3)

-   Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Rather than a histogram or
other meta information about the data, I was more interested in seeing the
actual images. So I wrote a routine to print slices of the images with their
proper names as given in signnames.csv. The results are pictured below.

[First 6 validation images][first_6_validation_images.png] [First 6 training
images][first_6_training_images.png]

This lead to me realize that the validation images and the training images were
not pre-shuffled. I had previosly attempted to train LeNet without shuffling the
images, and got accuracy scores of 0.07 or less. Shuffling the images was
determined to be critical to the process.

Also, doing a bincount on the images showed that some images had ten times as
many samples as others. This caused the CNN to over train on some images,
keeping accuracy low. I decided to even out the number of images by picking a
target number of images, duplicating images which were not well represented and
removing images that were over represented, so that each image type had the same
number of images. This gained me accuracy on the validation images of \> 0.93.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized the image data because testing showed that with
normalization accuracy increased approximately 5%.

As a second step, I decided to convert the images to grayscale.

Here is an example of a traffic sign image before and after grayscaling.

[BW Vs. Color][bwVsColor.png]

I decided to generate additional data by using the test data set provided.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer           | Description                               |
|-----------------|-------------------------------------------|
| Input           | 32x32x1 greyscale image                   |
| Convolution 5x5 | 1x1 stride, same padding, outputs 28x28x6 |
| RELU            |                                           |
| Max pooling     | 2x2 stride, outputs 14x14x6               |
| Convolution 5x5 | outputs 10x10x16                          |
| Relu            |                                           |
| Pooling layer   | output 5x5x16                             |
| Flatten         | output 400                                |
| Fully connected | output 120                                |
| Relu            |                                           |
| Fully connected | output 84                                 |
| Relu            |                                           |
| Fully Connected | output 43                                 |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a batch size of 25, which dramatically increased
accuracy.

Results can be found in the accompanying "accuracy_stats.txt" file.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

Training Accuracy = 0.998

Validation Accuracy = 0.934

Test Accuracy = 0.917

If an iterative approach was chosen: \* What was the first architecture that was
tried and why was it chosen? \* What were some problems with the initial
architecture? \* How was the architecture adjusted and why was it adjusted?

The architecture is basic LeNet from the last project. I did add a dropout
layer, but testing showed little effect. More testing and tuning might be of
assistance, since the testing was done prior to fitting the data.

\* Which parameters were tuned? How were they adjusted and why? \* What are some
of the important design choices and why were they chosen? For example, why might
a convolution layer work well with this problem? How might a dropout layer help
with creating a successful model?

If a well known architecture was chosen: \* What architecture was chosen? \* Why
did you believe it would be relevant to the traffic sign application? \* How
does the final model's accuracy on the training, validation and test set provide
evidence that the model is working well?

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I didn’t find appropriate images on the web. I used the test images, chosen at
random.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

My prediction: [ 9 7 13 10 9 31]

Actual values: [ 9 8 13 10 9 31]

picindex = 30 value = 9 name = No passing

picindex = 47 value = 7 name = Speed limit (100km/h)

picindex = 6 value = 13 name = Yield

picindex = 54 value = 10 name = No passing for vehicles over 3.5 metric tons

picindex = 30 value = 9 name = No passing

picindex = 1 value = 31 name = Wild animals crossing

The model was able to correctly guess 4 of the 5 traffic signs, which gives an
accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of
the Ipython notebook.
