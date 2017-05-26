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
3) Use color transforms, gradients, etc., to create a thresholded binary img
4) Apply perspective transform to rectify binary image (birds eye view)
5) Detect lane pixels and fit to find the lane boundary (try to use np.convolve)
6) Determine the curvature of the lane and vehicle position with respect to center
7) Warp the detected lane boundaries back onto the original image
8) Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
"""

def loadImage(fname):
    # cv2.imread is BGR
    img = mpimg.imread(fname)
    return img

def calibrateLens(nx, ny, folder):
    #prepare object points, like (0,0,0,), (1,0,0), (2,0,0)...
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store obj points and img points from all images
    objpoints = []
    imgpoints = []

    images = glob.glob(folder+'/calibration*.jpg')
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
    img = cv2.imread(folder+'/calibration5.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    print("Creating undistortion matrix.")

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite(folder+'/undistorted/test_undist.jpg',dst)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    print("Creating dist_pickle.p")
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist

    pickle.dump( dist_pickle, open( folder+"/dist_pickle.p", "wb" ) )
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

def undistortImage(img, mtx, dist):
    """
    Removes Lens distortion from RAW image. 
    Returns undistorted image
    """
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

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

def generate_binary_img(color_img, ksize = 3):
    image = cv2.cvtColor(color_img,cv2.COLOR_RGB2GRAY)

    # Choose a Sobel kernel size
    # ksize = 5 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # cv2.imwrite('thresh_kernel_tests/kernel'+str(ksize)+'.jpg',combined)

    return combined

def showImages(img1, img2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title('Modified img', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    return

def warp(img):
    img_size = (img.shape[1], img.shape[0])
    # Four source coordinates for realsense img
    src = np.float32(
        [[548,5],  # top right
        [621,471],        # bottom right
        [55,469],        # bottom left
        [122,23]])       # top left
    dst = np.float32(
        [[550,25],
        [550,475],
        [50,475],
        [50,25]])

    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped

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

testImg = loadImage("test_images/straight_lines1.jpg")
rs_img = loadImage("birds_eye.jpg")
# for i in range(3,19,2):
binary_img = generate_binary_img(testImg, ksize=7)

rs_warped = warp(rs_img)
rs_binary = generate_binary_img(rs_warped, ksize=7)

showImages(rs_warped, rs_binary)





