
# coding: utf-8

# In[1]:

# Matt London
# P4 - Advanced Lane Finding

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from Functions import *
"""
Design Plan:

1) Compute camera calibration matrix and distortion coeff given a set of chessboard images
2) Apply distortion correction to raw images
3) Use color transforms, gradients, etc., to create a thresholded binary img
4) Apply perspective transform to rectify binary image (birds eye view)
5) Detect lane pixels and fit to find the lane boundary (try to use np.convolve)
6) Determine the curvature of the lane and vehicle position with respect to center
7) Warp the detected lane boundaries back onto the original image
8) Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
"""

try:
    folder = "camera_cal"
    # Import pickle with calibrated distortion matrix
    dist_pickle = pickle.load( open( folder+"/dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    print(folder+" Pickle data loaded successfully")
except:
    # Create new pickle data based on camera calibration and distortion matricies
    calibrateLens(9,6,folder)


# # Load test Image to experiment with

# In[28]:

testImg = loadImage("test_images/test2.jpg")
warped, Minv = warp(testImg)
binary_img = generate_binary_img(testImg, ksize=7)
binary_warp = generate_binary_img(warped, ksize=7)
warped_binary, Minv = warp(binary_img)

showImages(testImg, binary_img,str("testImg"),str('binary_img'), testing=True)
showImages(warped, warped_binary,str("Warped Image"),str("Warped Binary"),testing=True)


# # Show image mask

# In[29]:

masked = mask(testImg)
showImages(testImg, masked)
testImg.shape


# # Visualize test images in various color spaces

# In[30]:

def hls_select(image, thresh=(90, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]    
    # 2) Apply a threshold to the S channel
    
    S_binary = np.zeros_like(S)
    L_binary = np.zeros_like(S)
    S_binary[(S> thresh[0]) & (S <= thresh[1])] = 1
    L_binary[(L> thresh[0]) & (L <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
#     show4images(image,H,L,S,"Original", 'H','L','S')
    return S_binary, L_binary, S, L

S_binary, L_binary, S , L = hls_select(testImg, thresh=(100, 255))
showImages(S_binary, L_binary,"S binary", "L binary")


# In[31]:

def rgb_select(image, thresh=(200,255)):
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    binary = np.zeros_like(R)
    binary[(R > thresh[0]) & (R <= thresh[1])] = 1
#     show4images(image,R,G,B,"Original", 'R','G','B')
    return binary, R

rgb_binary , R = rgb_select(testImg, thresh=(200,255))
showImages(testImg, rgb_binary, "Test Image", "(R)GB Binary")


# In[32]:

def hsv_select(image, thresh=(90,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H = hls[:,:,0]
    S = hls[:,:,1]
    V = hls[:,:,2]
    binary = np.zeros_like(S)
    binary[(V > thresh[0]) & (V <= thresh[1])] = 1
#     show4images(image,H,S,V,"Original", 'H','S','V')
    return binary

hsv_binary = hsv_select(testImg)
showImages(testImg, hsv_binary, 'Test Image', 'HS(V) Binary')


# In[33]:

def gray_select(img, thresh=(180,255)):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    binary = np.zeros_like(gray)
    binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1
    return binary
gray_binary = gray_select(testImg, thresh=(180,255))
showImages(testImg, gray_binary, 'Test Image', 'Gray Binary')


# # RGB threshold testing

# In[34]:

ksize = 5
gray = cv2.cvtColor(testImg,cv2.COLOR_RGB2GRAY)
gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(30, 100))
grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(30, 100))
mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(35, 110))
dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(.7,1.3 ))

test_combined = np.zeros_like(dir_binary)
test_combined[((gradx == 1)) & (grady == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1

show4images(gradx,grady,mag_binary, dir_binary, 'gradx','grady','mag_binary',"dir_binary")
showImages(gray,test_combined,'gradx', "combined")


# # Red threshold testing

# In[35]:

def gen_R_binary(R,ksize=5, testing=False):
    # ksize = 5
    gradx = abs_sobel_thresh(R, orient='x', sobel_kernel=ksize, thresh=(30, 100))
    grady = abs_sobel_thresh(R, orient='y', sobel_kernel=ksize, thresh=(30, 100))
    mag_binary = mag_thresh(R, sobel_kernel=ksize, mag_thresh=(35, 110))
    dir_binary = dir_threshold(R, sobel_kernel=ksize, thresh=(.7,1.3 ))

    r_combined = np.zeros_like(dir_binary)
    r_combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    if testing == True:
        show4images(gradx,grady,mag_binary, dir_binary, 'gradx','grady','mag_binary',"dir_binary")
    
    return r_combined
r_combined = gen_R_binary(R, testing=True)
showImages(R,r_combined,'gradx', "R combined")


# # S Threshold Testing

# In[36]:

def gen_S_binary(S,ksize=7, testing=False):
    # ksize = 7
    gradx = abs_sobel_thresh(S, orient='x', sobel_kernel=ksize, thresh=(30, 100))
    grady = abs_sobel_thresh(S, orient='y', sobel_kernel=ksize, thresh=(30, 100))
    mag_binary = mag_thresh(S, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(S, sobel_kernel=ksize, thresh=(.7,1.3 ))

    S_combined = np.zeros_like(dir_binary)
    S_combined[((gradx == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    if testing == True:
        show4images(gradx,grady,mag_binary, dir_binary, 'gradx','grady','mag_binary',"dir_binary")

    return S, S_combined
S, S_combined = gen_S_binary(S,testing=True)
showImages(S,S_combined,'S', "S combined")


# # Luminance Threshold Testing

# In[37]:

def gen_L_binary(L, ksize=7, testing=False):
#     ksize = 7
    gradx = abs_sobel_thresh(L, orient='x', sobel_kernel=ksize, thresh=(30, 100))
    grady = abs_sobel_thresh(L, orient='y', sobel_kernel=ksize, thresh=(30, 100))
    mag_binary = mag_thresh(L, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(L, sobel_kernel=ksize, thresh=(.7,1.3 ))

    L_combined = np.zeros_like(dir_binary)
    L_combined[((gradx == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    if testing == True:
        show4images(gradx,grady,mag_binary, dir_binary, 'gradx','grady','mag_binary',"dir_binary")
        
    return L_combined
L_combined = gen_L_binary(L, testing=True)
showImages(L,L_combined,'S', "combined L")


# In[38]:

binary_r = generate_binary_img(R)
binary_s = generate_binary_img(S)
showImages(R, binary_r, 'binary red', 'binary S')


# In[39]:

def combine_binary(img1, img2, flag='OR'):
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(img1)
    if flag == 'AND':
        combined_binary[(img1 == 1) & (img2 == 1)] = 1
    else:
        combined_binary[(img1 == 1) | (img2 == 1)] = 1
    return combined_binary

# LR_combined = combine_binary(rgb_binary, L_binary, flag='AND')
# combined = combine_binary(LR_combined,S_combined,flag='OR')

LR_combined = combine_binary(r_combined,L_combined)
combined = combine_binary(LR_combined,binary_s)
# showImages(binary_r,combined)
warped, Minv = warp(combined)
# showImages(mask(combined), warped)
show4images(S_combined, LR_combined, combined, warped,'S combined', 'LR Combined', 'LRS Combined', 'warped')


# # Find lane lines

# In[49]:

img = warped.astype(np.uint8)
histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
# plt.plot(histogram)
# plt.show()


# # Use sliding window algorithm to locate lane lines

# In[41]:

out_img, left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx = start_sliding_window(warped)

# plt.imshow(out_img)
# plt.plot(left_fitx, ploty, color='yellow')
# plt.plot(right_fitx, ploty, color='yellow')
# plt.xlim(0, 1280)
# plt.ylim(720, 0)
# plt.show()


# # Using the sliding windows we just created...

# In[42]:

result, left_fitx, right_fitx, ploty, leftx, rightx, left_fit, right_fit = continue_sliding_window(warped, left_fit, right_fit)
# plt.imshow(result)
# plt.plot(left_fitx, ploty, color='yellow')
# plt.plot(right_fitx, ploty, color='yellow')
# plt.xlim(0, 1280)
# plt.ylim(720, 0)
# plt.show()


# # Measuring Curvature

# In[43]:

# # Generate some fake data to represent lane-line pixels
# # ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
# quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
# # For each y position generate random x position within +/-50 pix
# # of the line base position in each case (x=200 for left, and x=900 for right)
# leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
#                               for y in ploty])
# rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
#                                 for y in ploty])

# leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
# rightx = rightx[::-1]  # Reverse to match top-to-bottom in y


# # Fit a second order polynomial to pixel positions in each fake lane line
# left_fit = np.polyfit(ploty, leftx, 2)
# left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
# right_fit = np.polyfit(ploty, rightx, 2)
# right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# Plot up the fake data
mark_size = 3
# plt.plot(left_fitx, ploty, 'o', color='red', markersize=mark_size)
# plt.plot(right_fitx, ploty, 'o', color='blue', markersize=mark_size)
# plt.xlim(0, 1280)
# plt.ylim(0, 720)
# plt.plot(left_fitx, ploty, color='green', linewidth=3)
# plt.plot(right_fitx, ploty, color='green', linewidth=3)
# plt.gca().invert_yaxis() # to visualize as we do the images
# plt.show()

# Define y-value where we want radius of curvature
# I'll choose the maximum y-value, corresponding to the bottom of the image
y_eval = np.max(ploty)
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
print(left_curverad, right_curverad)
# Example values: 1926.74 1908.48


# In[44]:

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


# In[45]:

def project_lane(img, mtx, dist, warped, left_fitx, right_fitx, ploty):
    undist = undistortImage(img, mtx, dist)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result
result = project_lane(testImg, mtx, dist, warped, left_fitx, right_fitx, ploty)
# plt.imshow(result)
# plt.show()


# # Line class to hold old information

# In[48]:

class VideoProcessor(object):

    # constructor function
    def __init__(self):
        # values of the last n fits of the line
        self.past_frames_left = []
        self.past_frames_right = []

        #polynomial coefficients averaged over the last n iterations
        self.best_fit_left = np.array([])
        self.best_fit_right = np.array([])

        self.first_frame = True

        self.mtx = np.array([])
        self.dist = np.array([])

    def moving_average_curvature(self):
        sum = 0
        N = len(self.past_frames_left)
        if N > 30:
            self.past_frames_left.pop(0)
            self.past_frames_right.pop(0)
            N = len(self.past_frames_left)

        l_coeff2 = 0
        l_coeff1 = 0
        l_coeff0 = 0

        r_coeff2 = 0
        r_coeff1 = 0
        r_coeff0 = 0

        for i in range(0,len(self.past_frames_left)):
            print(self.past_frames_left)
            # print(self.past_frames_left.shape)
            print(i)
            # n = 0
            # l_coeff[n] = i
            # n += 1
            l_coeff2 += self.past_frames_left[i][0]
            l_coeff1 += self.past_frames_left[i][1]
            l_coeff0 += self.past_frames_left[i][2]

            r_coeff2 += self.past_frames_right[i][0]
            r_coeff1 += self.past_frames_right[i][1]
            r_coeff0 += self.past_frames_right[i][2]

        # for i in self.past_frames_right:

        #     r_coeff2 += self.past_frames_right[i][0]
        #     r_coeff1 += self.past_frames_right[i][1]
        #     r_coeff0 += self.past_frames_right[i][2]

        l_coeff2 = l_coeff2 / N
        l_coeff1 = l_coeff1 / N
        l_coeff0 = l_coeff0 / N

        r_coeff2 = r_coeff2 / N
        r_coeff1 = r_coeff1 / N
        r_coeff0 = r_coeff0 / N


        self.best_fit_left = np.array([l_coeff2, l_coeff1, l_coeff0])
        self.best_fit_right = np.array([r_coeff2, r_coeff1, r_coeff0])

        return self.best_fit_left, self.best_fit_right

    def process_frame(self, frame):
        # your lane detection pipeline

        try:
            if self.first_frame == True:
                folder = "camera_cal"
                # Import pickle with calibrated distortion matrix
                dist_pickle = pickle.load( open( folder+"/dist_pickle.p", "rb"))
                self.mtx = dist_pickle["mtx"]
                self.dist = dist_pickle["dist"]
                print(folder+" Pickle data loaded successfully")
        except:
            # Create new pickle data based on camera calibration and distortion matricies
            calibrateLens(9,6,folder)
            print('Exception - calibrating lens')

        S_binary, L_binary, S , L = hls_select(frame, thresh=(100, 255))
        L_combined = gen_L_binary(L)
        rgb_binary , R = rgb_select(frame, thresh=(200,255))
        r_combined = generate_binary_img(R)
        LR_combined = combine_binary(r_combined, L_combined)
        binary_s = generate_binary_img(S)
        combined = combine_binary(LR_combined, binary_s)
        warped, Minv = warp(combined)
        warped = warped.astype(np.uint8)
        
        if self.first_frame == True:
            out_img, left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx = start_sliding_window(warped)

            # Update object attributes
            self.first_frame = False
            self.past_frames_left.append(left_fit)
            self.past_frames_right.append(right_fit)
            self.best_fit_left = left_fit
            self.best_fit_right = right_fit

            print('first frame')
        else:
            self.best_fit_left, self.best_fit_right = self.moving_average_curvature()
            result, left_fitx, right_fitx, ploty, leftx, rightx, left_fit, right_fit = continue_sliding_window(warped, self.best_fit_left, self.best_fit_right)

            self.past_frames_left.append(left_fit)
            self.past_frames_right.append(right_fit)
            # self.past_frames_left = left_fit
            # self.past_frames_right = right_fit

        ### Calculate curvature
        left_curverad, right_curverad = calc_curve(ploty, left_fitx, right_fitx, self.best_fit_left, self.best_fit_right)
        result = project_lane(frame, self.mtx, self.dist, warped, left_fitx, right_fitx, ploty)
        return result


# # Video Processing

# In[ ]:

VideoProcessor = VideoProcessor()
white_output = 'output_images/short_output.mp4'
clip1 = VideoFileClip("short_video.mp4")
white_clip = clip1.fl_image(VideoProcessor.process_frame) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)


# In[ ]:



