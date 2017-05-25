# Matt London
# P4 - Advanced Lane Finding

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

"""
Design Plan:

1) Compute camera calibration matrix and distortion coeff given a set of chessboard images
2) Apply distortion correction to raw images
3) Use color transforms, gradients, etc., t ocreate a thresholded binary img
4) Apply perspective transform to rectify binary image (birds eye view)
5) Detect lane pixels and fit to find the lane boundary (try to use np.convolve)
6) Determine the curvature of the lane and vehicle position with respect to center
7) Warp the detected lane boundaries back onto the original image
8) Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
"""

def loadImage(fname):
	# cv2.imread is BGR
	img = cv2.imread(fname)
	return img

def calibrateLens(nx, ny):
	#prepare object points, like (0,0,0,), (1,0,0), (2,0,0)...
	objp = np.zeros((ny*nx, 3), np.float32)
	objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

	# Arrays to store obj points and img points from all images
	objpoints = []
	imgpoints = []

	images = glob.glob('camera_cal/calibration*.jpg')
	print("Reading in images. ")

	# Step through the list and search for chessboard corners
	for idx, fname in enumerate(images):
	    img = cv2.imread(fname)
	    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	    # Find the chessboard corners
	    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

	    # If found, add object points, image points
	    if ret == True:
	        objpoints.append(objp)
	        imgpoints.append(corners)

	        # Draw and display the corners
	        cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
	        # write_name = 'corners_found'+str(idx)+'.jpg'
	        # cv2.imwrite(write_name, img)
	        cv2.imshow('img', img)
	        cv2.waitKey(500)
	    print("Image {}...\n".format(idx))
	cv2.destroyAllWindows()

	# Test undistortion on an image
	img = cv2.imread('camera_cal/calibration5.jpg')
	img_size = (img.shape[1], img.shape[0])

	# Do camera calibration given object points and image points
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

	print("Creating undistortion matrix.")

	dst = cv2.undistort(img, mtx, dist, None, mtx)
	cv2.imwrite('camera_cal/undistorted/test_undist.jpg',dst)

	# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
	print("Creating dist_pickle.p")
	dist_pickle = {}
	dist_pickle["mtx"] = mtx
	dist_pickle["dist"] = dist
	pickle.dump( dist_pickle, open( "camera_cal/dist_pickle.p", "wb" ) )
	#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

	# Visualize undistortion
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
	ax1.imshow(img)
	ax1.set_title('Original Image', fontsize=30)
	ax2.imshow(dst)
	ax2.set_title('Undistorted Image', fontsize=30)
	return

	# gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	# # Find chessboard corners for 9x6 board
	# ret, corners = cv2.findChessboardCorners(gray,(nx,ny), None)
	# #Draw detected corners on image
	# img = cv2.drawChessboardCorners(img,(nx,ny),corners, ret)
	# # Camera calib, given obj points, image points, and the shape of the grayscale image
	# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	# # Undistorting Image
	# dst = cv2.undistort(img, mtx, dist, None, mtx)
	# undist = cv2.undistort(img, mtx, dist, None, mtx)

def undistortImage(img):
	"""
	Removes Lens distortion from RAW image 
	"""
	
	# Import pickle with calibrated distortion matrix
	dist_pickle = pickle.load( open( "camera_cal/dist_pickle.p", "rb"))
	mtx = dist_pickle["mtx"]
	dist = dist_pickle["dist"]

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

if __name__ == "__main__":

	calibrateCamera(9,6)
