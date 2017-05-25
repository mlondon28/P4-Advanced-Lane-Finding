# Matt London
# P4 - Advanced Lane Finding

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

