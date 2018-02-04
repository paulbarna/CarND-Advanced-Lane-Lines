import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib qt

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')


# Step through the list and search for chessboard corners
for N,fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
        

cv2.destroyAllWindows()

# Read in an image
img = cv2.imread('test_images/test3.jpg')

# TODO: Write a function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera and cv2.undistort()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist,mtx,dist

undistorted,mtx,dist = cal_undistort(img, objpoints, imgpoints)

undistorted=cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# Read in an image
imgCal = cv2.imread('test_images/straight_lines1.jpg')
imgRGB = cv2.cvtColor(imgCal, cv2.COLOR_BGR2RGB)
imgUndist = cv2.undistort(imgRGB, mtx, dist, None, mtx)


h,w = imgUndist.shape[:2]

src = np.float32([(534,492),
                  (750,488), 
                  (240,684), 
                  (1056,678)])

dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])


def corners_unwarp(img, src, dst):
    # Pass in your image into this function
    # Write code to do the following steps
    # Undistort using mtx and dist
    h,w = img.shape[:2]
    M = cv2.getPerspectiveTransform(src, dst)
    Minv=cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, Minv

top_down, perspective_M = corners_unwarp(imgUndist, src,dst)

hWarped,wWarped= imgUndist.shape[:2]
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(imgRGB)
x = [src[0][0],src[2][0],src[3][0],src[1][0],src[0][0]]
y = [src[0][1],src[2][1],src[3][1],src[1][1],src[0][1]]
xWarped = [dst[0][0],dst[2][0],dst[3][0],dst[1][0],dst[0][0]]
yWarped = [dst[0][1],dst[2][1],dst[3][1],dst[1][1],dst[0][1]]
ax1.plot(x, y, color='#FF0000', alpha=0.5, linewidth=2, solid_capstyle='round', zorder=2)
ax1.set_ylim([h,0])
ax1.set_xlim([0,w])
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.plot(xWarped, yWarped, color='#FF0000', alpha=0.5, linewidth=2, solid_capstyle='round', zorder=2)
ax2.set_ylim([hWarped,0])
ax2.set_xlim([0,wWarped])
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient=='x', orient=='y')
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    binary_output = sxbinary # Remove this line
    return binary_output
    
# Run the function
grad_binary = abs_sobel_thresh(top_down, orient='x', thresh_min=20, thresh_max=100)
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(top_down)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(grad_binary, cmap='gray')
ax2.set_title('Thresholded Gradient', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Calculate the magnitude 
    mag_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    # 5) Create a binary mask where mag thresholds are met
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    binary_output = np.copy(sxbinary) 
    return binary_output
    
# Run the function
mag_binary = mag_thresh(top_down, sobel_kernel=3, mag_thresh=(30, 100))
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(top_down)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(mag_binary, cmap='gray')
ax2.set_title('Thresholded Magnitude', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output =  np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    binary_output = np.copy(binary_output) # Remove this line
    return binary_output
    
# Run the function
dir_binary = dir_threshold(top_down, sobel_kernel=15, thresh=(0.7, 1.3))
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(top_down)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(dir_binary, cmap='gray')
ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(hls[:,:,2])
    binary_output[(hls[:,:,2] > thresh[0]) & (hls[:,:,2] <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    #binary_output = np.copy(img) # placeholder line
    return binary_output
    
hls_binary = hls_select(top_down, thresh=(180, 255))

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(top_down)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(hls_binary, cmap='gray')
ax2.set_title('Thresholded S', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

def lab_select(img, thresh):
    # 1) Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab_b = lab[:,:,2]
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    # 3) Return a binary image of threshold result
    return binary_output

lab_binary = lab_select(top_down, thresh=(145, 200))

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(top_down)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(lab_binary, cmap='gray')
ax2.set_title('Thresholded B channel', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

def luv_select(img, thresh):
    # 1) Convert to LUV color space
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(luv[:,:,0])
    binary_output[(luv[:,:,0] > thresh[0]) & (luv[:,:,0] <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    #binary_output = np.copy(img) # placeholder line
    return binary_output
    
luv_binary = luv_select(top_down, thresh=(215, 255))

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(top_down)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(luv_binary, cmap='gray')
ax2.set_title('Thresholded L channel', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# Edit this function to create your own pipeline.

def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    
    # Generate binary thresholded images
    b_channel = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:,:,2]
    l_channel = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)[:,:,0]  
    
    # Set the upper and lower thresholds for the b channel
    b_thresh_min = 145
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1
    
    # Set the upper and lower thresholds for the l channel
    l_thresh_min = 215
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    combined_binary = np.zeros_like(b_binary)
    combined_binary[(l_binary == 1) | (b_binary == 1)] = 1
    
    return combined_binary


testImg = cv2.imread('test_images/test3.jpg')
testImgRGB = cv2.cvtColor(testImg, cv2.COLOR_BGR2RGB)
testImgUndist = cv2.undistort(testImgRGB, mtx, dist, None, mtx)

test_h,test_w = testImgUndist.shape[:2]

testSrc = np.float32([(576,478),
                  (754,478), 
                  (311,657), 
                  (1048,657)])

testDst = np.float32([(450,0),
                  (test_w-450,0),
                  (450,test_h),
                  (test_w-450,test_h)])
test_top_down, test_perspective_M = corners_unwarp(testImgUndist, testSrc,testDst)
binary_warped = pipeline(test_top_down)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
f.tight_layout()

ax1.imshow(testImgRGB)
ax1.set_title('Original Image', fontsize=30)

ax2.set_title('Combined S channel and gradient thresholds',fontsize=30)
ax2.imshow(binary_warped, cmap='gray')
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

def slidingWindowPart1(binary_warped):  
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 90
    # Set minimum number of pixels found to recenter window
    minpix = 45
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.array([0,0,0], dtype='float') 
    right_fit = np.array([0,0,0], dtype='float') 
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
        
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img, left_fit, right_fit, left_lane_inds, right_lane_inds


out_img, left_fit, right_fit, left_lane_inds, right_lane_inds= slidingWindowPart1(binary_warped)


 # Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


# At this point, you're done! But here is how you can visualize the result as well:
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
f.tight_layout()

ax1.imshow(testImgRGB)
ax1.set_title('Original Image', fontsize=30)

ax2.set_title('sliding windows',fontsize=30)
ax2.imshow(binary_warped, cmap='gray')
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)

def slidingWindowPart2(binary_warped,left_fit,right_fit):
        
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 90
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each  
    left_fit_current=np.array([0,0,0], dtype='float') 
    right_fit_current=np.array([0,0,0], dtype='float') 
    if len(leftx) != 0:
        left_fit_current= np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit_current= np.polyfit(righty, rightx, 2)
        
    # Create an image to draw on 
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
   
    return out_img, left_fit_current, right_fit_current, left_lane_inds, right_lane_inds

out_img, left_fit, right_fit, left_lane_inds, right_lane_inds = slidingWindowPart2(binary_warped,left_fit,right_fit)

# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# Create an image to show the selection window
window_img = np.zeros_like(out_img)
margin=80

# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# Draw the lane onto the warped blank image
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
f.tight_layout()

ax1.imshow(testImgRGB)
ax1.set_title('Original Image', fontsize=30)

ax2.set_title('sliding windows once we know where the line is',fontsize=30)
ax2.imshow(binary_warped, cmap='gray')
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

plt.imshow(result)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)

def MeasuringCurvatureAndDistance(binary_warped, left_fit, right_fit,left_lane_inds,right_lane_inds):
    
    left_curverad, right_curverad, distance = (0, 0, 0)
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    h = binary_warped.shape[0]
    w = binary_warped.shape[1]
    ploty = np.linspace(0, h-1, num=h)
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(leftx) != 0 and len(rightx) != 0:
        # fit polynomials
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # calculate the radius of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # calculate the distance by subtracting the mean of left fit and right fit intercepts, from the image weight median 
    if right_fit is not None and left_fit is not None:
        left_fit_intercepts = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        right_fit_intercepts = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        distance = ((w/2) - ((right_fit_intercepts + left_fit_intercepts)/2)) * xm_per_pix
        #print (distance)
    return left_curverad, right_curverad, distance

def Drawing(img, binary_warped, left_fit, right_fit, Minv,left_curverad,right_curverad,distance):
    img = np.copy(img)
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    h,w = binary_warped.shape[:2]
    ploty = np.linspace(0, h-1, num=h)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (binary_warped.shape[1], binary_warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Radius of Curvature: ' + '{:04.2f}'.format((left_curverad+right_curverad)/2) + 'm'
    cv2.putText(result, text, (50,100), font, 2, (200,255,155), 2, cv2.LINE_AA)
    direction = ''
    if distance > 0:
        direction = 'right'
    elif distance < 0:
        direction = 'left'
    text = 'Vehicle is '+'{:04.5f}'.format(abs(distance)) + 'm ' + direction + ' of center'
    cv2.putText(result, text, (50,150), font, 2, (200,255,155), 2, cv2.LINE_AA)
    
    #resultRGB = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
  
    return result
	
left_curverad_meas,right_curverad_meas,distance_meas= MeasuringCurvatureAndDistance(binary_warped, left_fit, right_fit,left_lane_inds,right_lane_inds)

Test_Img_Drawing = Drawing(testImg, binary_warped, left_fit, right_fit, test_perspective_M,left_curverad_meas,right_curverad_meas,distance_meas)

Test_Img_Drawing=cv2.cvtColor(Test_Img_Drawing, cv2.COLOR_BGR2RGB)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
f.tight_layout()

ax1.imshow(testImgRGB)
ax1.set_title('Original Image', fontsize=30)

ax2.set_title('Original (undistorted) image with lane area drawn',fontsize=20)
ax2.imshow(Test_Img_Drawing)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.array([0,0,0], dtype='float')   
        #polynomial coefficients for the most recent fit
        self.current_fit =[]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #buffer length of stored fits
        self.FramesBufferN = 4
        #margin diff between fits in pixels
        self.PixDiff= 100.
    
    def TrackingAnalysis(self, frame_fit):
        if frame_fit is None:
            if len(self.current_fit) > 0:
                # drop the last frame fit if frame fit is None
                self.current_fit = self.current_fit[:len(self.current_fit)-1]
                if len(self.current_fit) > 0:
                # get the average for the existing fits and assign it to the best_fit
                    self.best_fit = np.average(self.current_fit, axis=0)
            self.detected = False
        else:
            if self.best_fit is not None:
                # compute the diff between best fit and the frame fit, if it exceeds N pixels margin, the frame fit is droped
                self.diffs = abs(frame_fit-self.best_fit)
            if (self.diffs[2] > self.PixDiff) and len(self.current_fit) > 0:
                self.detected = False
            else:
                self.current_fit.append(frame_fit)
                if len(self.current_fit) > self.FramesBufferN:
                    # drop off the old frames
                    self.current_fit = self.current_fit[len(self.current_fit)-self.FramesBufferN:]
                self.best_fit = np.average(self.current_fit, axis=0)
                self.detected = True
            
def image_processing(image):
    image = np.copy(image)

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_undist = cv2.undistort(image, mtx, dist, None, mtx)
    h = image_undist.shape[0]
    w = image_undist.shape[1]
    margin=100

    top_down, perspective_Minv = corners_unwarp(image_undist, testSrc,testDst)
    binary_warped = pipeline(top_down)
    
    if not left_lane.detected or not right_lane.detected:
        #fit the polynomial to the binary image
        out_img, left_fit, right_fit, left_lane_inds, right_lane_inds = slidingWindowPart1(binary_warped)
    else:
        #fit the polynomial to the binary image by using the previous fit
        out_img, left_fit, right_fit, left_lane_inds, right_lane_inds = slidingWindowPart2(binary_warped,left_lane.best_fit,right_lane.best_fit)
    
    
    if left_fit is not None and right_fit is not None:
        left_fit_intercepts = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        right_fit_intercepts = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        #if the intercepts at the bottom of the image exceed the margin (in pixels) discard the fits
        if abs(h/2 - abs(left_fit_intercepts-right_fit_intercepts)) > margin:
                left_fit,right_fit = (None,None)
                
        
    #run both lanes through analysis to compare the frame fit to the existing history    
    left_lane.TrackingAnalysis(left_fit)
    right_lane.TrackingAnalysis(right_fit)
    
    # draw if there is a fit
    if left_lane.best_fit is not None and right_lane.best_fit is not None:
        left_curverad_meas,right_curverad_meas,distance_meas= MeasuringCurvatureAndDistance(binary_warped, left_lane.best_fit, right_lane.best_fit,left_lane_inds,right_lane_inds)
        Img_Drawing = Drawing(image, binary_warped, left_lane.best_fit, right_lane.best_fit, perspective_Minv,left_curverad_meas,right_curverad_meas,distance_meas)
    #leave the image empty otherwise    
    else:
        Img_Drawing = image
    
    return Img_Drawing

	from moviepy.editor import VideoFileClip

video_output_project = 'project_video_output_final.mp4'
video_input_project = VideoFileClip('project_video.mp4')

# apply src and dst coordinates per image characteristics/camera & lanes position 
video_input_project.save_frame("project_video.jpg")
testImg = cv2.imread('project_video.jpg')
testImgRGB = cv2.cvtColor(testImg, cv2.COLOR_BGR2RGB)
image_undist = cv2.undistort(testImgRGB, mtx, dist, None, mtx)
h,w = image_undist.shape[:2]

testSrc = np.float32([(576,478),
                  (754,478), 
                  (311,657), 
                  (1048,657)])

testDst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])

#initialize the lane statistics 
left_lane = Line()
right_lane = Line()


video_result = video_input_project.fl_image(image_processing)
%time video_result.write_videofile(video_output_project, audio=False)




