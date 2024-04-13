import numpy as np
import cv2
from matplotlib import pyplot as plt


# 1. Compute disparity between the two stereo images.
def ShowDisparity(img_left, img_right, bSize=5):
    # Initialize the stereo block matching object
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=bSize)

    # Compute the disparity image
    disparity = stereo.compute(img_left, img_right)

    # Normalize the image for representation
    min = disparity.min()
    max = disparity.max()
    disparity = np.uint8(255 * (disparity - min) / (max - min))
    # Plot the result
    plt.imshow(disparity, 'gray')
    plt.show()
