## Vehicle Detection Project

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
    * Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
    * Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

All of the code for the project is contained in the Jupyter notebook `vehicle_detection_final.ipynb` 

---

### Data Exploration

All the vehicle and non-vehicle data paths are initially loaded from the input directories, leading to the following counts:

```
Number of Non-Vehicles:  8968
Number of Vehicles:  8792
```

An example of a car image and not car image is shown below:

![car](output_images/car.jpg)

![car](output_images/not_car.jpg)

Next, let us explore these images to determine the best color transform to use (if needed). Multiple color spaces have been explored in the notebook, by plotting color and comparing the difference of these histograms across a randomly selected car and non-car image.

Multiple images were explored, with YUV chosen as the final color space.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

HOG feature extraction is defined by the method `get_hog_features` and is contained in the cell titled HOG Parameters"  The figure below shows a comparison of a car image and its associated histogram of oriented gradients, as well as the same for a non-car image.

![car4](output_images/car4.jpg)

![hog_car8](output_images/hog_car8.jpg)





![not_car4](output_images/not_car4.jpg)

![hog_notcar8](output_images/hog_notcar8.jpg)



The best parameters for differentiation have been estimated accordingly as 

- Orient = 11
- Pix per cell = 16
- Cells per block = 2

The method `extract_features` in the section titled "Extract Features" accepts a list of image paths and HOG parameters and produces a flattened array of HOG features for each image in the list.

Next, the section titled "Setup Parameters" defines parameters for HOG feature extraction and extract features for the entire dataset. These feature sets are combined and a label vector is defined (`1` for cars, `0` for non-cars). The features and labels are then shuffled and split into training and test sets in preparation to be fed to a linear support vector machine (SVM) classifier. 



#### 2. Explain how you settled on your final choice of HOG parameters.

The final choice of HOG parameters is based upon the accuracy performance of the SVM classifier, after multiple trials to reach 0.9848

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The section titled "Train Linear SVM Classifier" I trained a linear SVM with the default classifier parameters and using HOG features alone (I did not use spatial intensity or channel intensity histogram features) and was able to achieve a test accuracy of 98.17%. 

---

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In the section titled "Search and Classify" the method `find_cars` from the lesson materials is used. The method combines HOG feature extraction with a sliding window search, but rather than perform feature extraction on each window individually which can be time consuming, the HOG features are extracted for the entire image (or a selected portion of it) and then these full-image features are subsampled according to the size of the window and then fed to the classifier. The method performs the classifier prediction on the HOG features for each window region and returns a list of rectangle objects corresponding to the windows that generated a positive ("car") prediction.

Several configurations of window sizes and positions, with various overlaps in the X and Y directions were explored Essentially this is a trial and error process. 

The final algorithm calls `find_cars` for each window scale and the rectangles returned from each method call are aggregated. An overlap of 75% in the Y direction and 50% in the X direction) produced more redundant true positive detections, which were preferable given the heatmap strategy described below. Additionally, only an appropriate vertical range of the image is considered for each window size (e.g. smaller range for smaller scales) to reduce the chance for false positives in areas where cars at that scale are unlikely to appear. 

The image below shows the rectangles returned by `find_cars` drawn onto one of the test images in the final implementation. Notice that there are several positive predictions on each of the near-field cars, and one positive prediction on a car in the oncoming lane.

![07_all_detections](output_images/07_all_detections.png)
The `add_heat` function increments the pixel value (referred to as "heat") of an all-black image the size of the original image at the location of each detection rectangle. Areas encompassed by more overlapping rectangles are assigned higher levels of heat. The following image is the resulting heatmap from the detections in the image above:

A threshold is applied to the heatmap (in this example, with a value of 1), setting all pixels that don't exceed the threshold to zero. The result is below:

![08_heatmap](output_images/08_heatmap.png)

The `scipy.ndimage.measurements.label()` function collects spatially contiguous areas of the heatmap and assigns each a label:

![10_label_heatmap](output_images/10_label_heatmap.png)

And the final detection area is set to the extremities of each identified label:

![11_final_boxes](output_images/11_final_boxes.png)

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The results of passing all of the project test images through the above pipeline are displayed in the images below:

![12_all_test_detects](output_images/12_all_test_detects.png)


Optimization techniques included changes to window sizing and overlap, and lowering the heatmap threshold to improve accuracy of the detection (higher threshold values tended to underestimate the size of the vehicle).

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to video](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for processing frames of video is contained in the cell titled "Video Processing" and "Process Video" is identical to the code for processing a single image described above, with the exception of storing the detections (returned by `find_cars`) from the previous 15 frames of video using the `prev_rects` parameter from a class called `Vehicle_Detect`. Rather than performing the heatmap/threshold/label steps for the current frame's detections, the detections for the past 15 frames are combined and added to the heatmap 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems in this project were mainly concerned with detection accuracy. Balancing the accuracy of the classifier with execution speed was crucial. 

The pipeline is probably most likely to fail in cases where 

1. vehicles don't resemble those in the training dataset
2. lighting and environmental condition differences 
3. Oncoming cars are an issue, as well as distant cars could cause misclassification

The algorithm could be made more robust by using a convolutional neural network - YOLO or SSD were not tried due to lack of time