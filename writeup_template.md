## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/UndistortedIMG.jpeg "Undistorted"
[image2]: ./output_images/UndistortedWarpedIMG.jpeg "Undistorted and Warped Image"
[image3]: ./output_images/SlidingWindow.jpeg "Sliding Window"
[image4]: ./output_images/SlidingWindowPreviousFit.jpeg "Sliding Window using previous fit"
[image5]: ./output_images/LaneAreaDrawn.jpeg "Fit Visual"
[image6]: ./output_images/ThresholdedBchannel.jpeg "Thresholded B channel"
[image7]: ./output_images/ThresholdedLchannel.jpeg "Thresholded L channel"
[image8]: ./output_images/UndistortedTestIMG.jpeg "Undistorted Image"
[image9]: ./output_images/CombinedSchannel_GradientThreshold.jpeg "Gradient and S Channel Threshold"
[video1]: ./output_images/project_video_output_final.mp4 "Video"
[video2]: ./output_images/challenge_video_output_final.mp4 "Video Challenge"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" .  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image8]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I experimented with the Sobel Operator, the magnitude of the gradient, direction of the gradient, HLS, Lab, LUV thresholds to generate binary images and compare the results (which appears starting with the 3rd code cell up to the 8th code cell of the IPython notebook).  Here's an example of my output for this step.

![alt text][image9]

However for the video processing I chose a combined Lab And Luv color space threshold (b channel and l channel) which proved to generate a better binary image. (which appears in the 10th code cell of the IPython notebook`)

![alt text][image6]
![alt text][image7]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `corners_unwarp()`, which appears in the 3rd code cell of the IPython notebook.  The `corners_unwarp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

The source was manually adjusted for each video in order to obtain better accuracy when fitting the polynomial.

drc = np.float32([(576,478),
                  (754,478), 
                  (311,657), 
                  (1048,657)])

dst = np.float32([(450,0),
                  (image.shape[1]-450,0),
                  (450,image.shape[0]),
                  (image.shape[1]-450,image.shape[0])])


| Source        | Destination   | 
|:-------------:|:-------------:| 
| 576, 478      | 450, 0        | 
| 754, 478      | 830, 0        |
| 311, 657      | 450, 720      |
| 1048, 657     | 830, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image2]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image3]
![alt text][image4]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the 13th code cell of the IPython notebook.  

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the 14th code cell of the IPython notebook in the function `Drawing()`.  Here is an example of my result on a test image:

![alt text][image5]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).


Here's a [link to my project_video result](./output_images/project_video_output_final.mp4)
Here's a [link to my challenge_video result](./output_images/challenge_video_output_final.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The techniques I used were mainly described within the lecture. The one step which proved to be critical for generating the binary image, are the combined Lab And Luv color space thresholds(b and l channels).
It proved to be particularly tricky to get the thresholds right as well.
I chose to hardcode the src and dst points for the perspective transform, by manually obtaining the src points based on the first frame lanes position.
However this is an aspect where I can improve by automatically updating the src and dst points each frame. This could improve the performance of the lane detection for the harder challenge video.
For measuring the curvature I used the lecture example while for calculating the position of the vehicle with respect to center I subtracted the mean value of the left and right fit intercepts from the image weight median.
In terms of buffering the tracking statistics I used a line class where I checked each frame fit against the best fit, while storing the last 5 fits for both the left and right lanes. 
I also dropped the frame fit if the lane is not detected, just to make sure the buffer average is not affected. Additionally I included a check where out of range intercepts at the bottom of the image would discard the fits as well.
Also an adaptive buffer storing functionality would provide better performance for the wobbly roads while an online src points calculation would improve the polynomial fit as well.
