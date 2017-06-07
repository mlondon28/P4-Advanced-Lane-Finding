
# **Advanced Lane Finding** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road


[//]: # (Image References)

[image1]: ./images/undistort_example.png "Original Image / Undistorted Image"
[image2]: ./images/hls_binary.png "HLS Binary"
[image3]: ./images/R_sobels.png "Sobel Filters on Red Channel"
[image4]: ./images/R_thresh.png "Red Sobel Threshold"
[image5]: ./images/L_sobel.png "Luminance Sobel Threshold"
[image6]: ./images/S_sobel.png "Saturation Sobel Threshold"
[image7]: ./images/ycrcb_binary.png "YCrCb Binary"
[image8]: ./images/combined.png "Combined Binary"
[image9]: ./images/histogram.png "Histogram of Lane Lines"
[image10]: ./images/sliding_window_plt.png "Sliding Window"
[image11]: ./images/identified_lanes_warped.png "Identified Lanes"
[image12]: ./images/radius_equation.png "Radius Equation"
[image13]: ./images/projected_lane.png "Projected Lane"
[image14]: ./Videos/short_output11.gif "Output Gif"


---

### Reflection

## Camera Calibration

To calibrate the lens used for all of the images in the project videos I used a few OpenCV techniques and functions to generate the correct camera matrix and distortion coefficients. 

The process I used was:

1) Iterate through images of a chessboard (taken at different angles / distances) taken with the same camera lens used on the dash cam videos

2) Find chessboard corners based on the number of x and y inside corners (Used `cv2.findChessboardCorners()`). 

3) Draw chessboard corners on original image to verify all the corners were found correctly. 

4) Then using `cv2.calibrateCamera()` to return the camera matrix (`mtx`) and distortion (`dist`) coefficients

5) `mtx` and `dist` were then used in conjunction with `cv2.undistort()` to return an undistorted image

Once this was done, the camera matrix (mtx) and distortion coefficients (dist) were saved in a pickle so they could be loaded later without having to recalculate.



**Here is a test image unedited and then undistorted:**
![alt text][image1]

```python
def undistortImage(img, mtx, dist):
    """
    Removes Lens distortion from RAW image. 
    Returns undistorted image
    """
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
```


## Image Pipeline 

#### Correcting for Image Distortion

The procedure listed above in the camera calibration section was used in the pipeline to correct for lens distortion.

#### Creating a Binary Image

There are many filters that went into creating the final binary image. The final binary needed to be robust enough to successfully identify lines in rapidly changing light and shadow as well as different colored lane lines and pavement. 

Color spaces used:
- RGB: (Red, Green, Blue). The Red channel was used to help identify yellow lines via a Sobel filter. ![Red Sobels][image3] ![Red combined][image4]
- HLS: (Hue, Lightness, Saturation). Lightness and saturation were separated, thresholded and used for the final binary image ![HLS][image2]
- Luminance Sobel threshold: HLS Luminance channel was used to help pull out lane lines in various (shaded) lighting conditions. ![Luminance Sobel][image5]
- Saturation Sobel Threshold: HLS Saturation channel was used to pull out colored lines in all lighting situations. ![S_sobel][image6]
- YCrCb Binary: To make the yellow lane detection in all lighting situations more robust, the Cb channel was used. ![YCrCb Binary][image7]
- Finally all channels were combined and masked to yield the final warped binary image. ![Combined Image][image8]

The Sobel Theshold functions that were used: 

```python
def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    sobelx = cv2.Sobel(image, cv2.CV_64F,1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F,0,1, ksize=sobel_kernel)
    mag = np.sqrt(sobelx**2 + sobely**2)
    scaled = np.uint8(255*mag/np.max(mag))
    mag_binary = np.zeros_like(scaled)
    mag_binary[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    sobelx = cv2.Sobel(image, cv2.CV_64F,1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F,0,1, ksize=sobel_kernel)
    absx = np.absolute(sobelx)
    absy = np.absolute(sobely)
    grad_dir = np.arctan2(absy, absx)
    dir_binary = np.zeros_like(grad_dir)
    dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    return dir_binary
    
def generate_binary_img(image, ksize = 7):
    # Choose a Sobel kernel size
    # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(30, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(30, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined
```

#### Transforming the Image Perspective

For image transformation `cv2.warpPerspective()` was used after the source and destination points were designated and a perspective transformation matrix was created via `cv2.getPerspectiveTransform(src,dst)`. 

The Warp function that was used is below:

```python
def warp(img):
    img_size = (img.shape[1], img.shape[0])
    # Four source coordinates for img
    xmax = img.shape[1]
    ymax = img.shape[0]
    offset = 200
    
    d_top_left = [200,0]
    d_top_right = [1000,0]
    d_bottom_right = [1000,720]
    d_bottom_left = [200,720]
    
    top_left = [563,471]
    top_right = [714,471]
    bottom_right = [1090,720]
    bottom_left = [221,720]
    
    src = np.float32(
        [top_right,     # top right
        bottom_right,   # bottom right
        bottom_left,    # bottom left
        top_left])      # top left
    dst = np.float32(
        [[d_top_right],     # top right
        [d_bottom_right],   # bottom right
        [d_bottom_left],    # bottom left
        [d_top_left]])      # top left

    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, Minv
```

*An example of a warped image can be seen above*

#### Identifying Lane-Line Pixels

I used a histogram and sliding window approach to identify lane lines. 

![Lane Histogram][image9]

This is the histogram of the warped image. Using this method allows us to easily generalize where the lane lines are assuming the conversion to a binary is good enough quality. 

Then a sliding window algorithm is applied and a second order polynomial is fit to the detected pixels. The estimated lane line is shown below.

![Sliding Window 1][image10]

Once we've detected the lanes via the sliding window, we don't need restart the search for the lanes from scratch as we can use the previously discovered sliding windows as starting points to look for the lanes. 

At this point, using the margin of the windows, we can continue identifying lanes. Lane identification based on previously found lanes is shown below:

![Sliding Window 2][image11]


#### Calculating Road Curvature and Center of Lane

Now that we have two polyfit lines approximating the lane lines we can calculate the center of the lane and the road curvature based on the position of the camera and the projected curve respectively. 

To get the radius of each lane this equation was used: ![Radius Equation][image12]

##### Below is the function used to calculate the radius of the curve and convert the radii from pixels to meters. 

```python
def calc_curve(ploty, left_fitx, right_fitx, left_fit, right_fit):
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad

left_curverad, right_curverad = calc_curve(ploty, left_fitx, right_fitx, left_fit, right_fit)
print(left_curverad, 'm', right_curverad, 'm')
# Example values: 632.1 m    626.2 m
```

##### This is the function used to calculate the car position with respect to the center of the lane.
This works by taking the midpoints of the two lines that the histogram finds and then calculating how many pixels off center the camera is from the center of the lines.

```python
def car_position(img):
        hist = np.sum(img[img.shape[0]//2:,:], axis=0)
        midpt = np.int(hist.shape[0]/2)
        camera_position = img.shape[1]/2
        left_x_predictions = np.argmax(hist[:midpt])
        right_x_predictions = np.argmax(hist[midpt:]) + midpt
        lane_center = (right_x_predictions + left_x_predictions)/2
        center_offset_pix = abs(camera_position - lane_center)
        location_str = "Vehicle dist. from center: " + str(center_offset_pix)
        return location_str
    
car_position(img)
```

#### Final Product

The final image result from the pipeline has the projected lane that is based on the detected lane lines and is shown below in green.

![Projected Lane][image13]

## Video Pipeline

![Output Gif][image14]

## Discussion

The video pipeline described above did a great job detecting lane lines in various different road and lighting conditions. 

Some potential shortcomings is the jitter at edges. This could be improved by added in better logic to throw out bad curvature calculations and using previously approved polynomial fits. 

I would also continue adding in different threshold gradients to make the detection more robust. I would be interested to see what it would take to make this pipeline perform well at night.
