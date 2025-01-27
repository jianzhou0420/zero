import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# Load the image
import re


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


father_path = '/home/jian/Downloads/results-seed-1357-002/results-ddim-sampling_steps-20-seed-1357/epoch-999/images'
image_lists = sorted(os.listdir(father_path), key=natural_sort_key)


for image_name in image_lists:
    image_path = os.path.join(father_path, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    canny_edges = cv2.Canny(image, 120, 150)

    kernel = np.ones((3, 3), np.uint8)
    canny_edges = cv2.dilate(canny_edges, kernel, iterations=1)

    # cv2.imshow('Original', image)
    # cv2.imshow('Canny Edges', canny_edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # Threshold the image (binarization)
    # _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    # binary = cv2.bitwise_not(binary)

    # # Define a kernel for erosion
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # # Iterative erosion
    # eroded = binary.copy()
    # for i in range(1):  # Number of iterations controls how much the image shrinks
    #     eroded = cv2.erode(eroded, kernel)
    # # cv2.imshow('Original', binary)
    # # cv2.imshow('Eroded', eroded)
    # # print(kernel)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # # # Skeletonization (Optional)

    def skeletonize(img):
        skeleton = np.zeros(img.shape, np.uint8)
        temp = np.zeros(img.shape, np.uint8)
        eroded = img.copy()
        while cv2.countNonZero(eroded) > 0:
            # Perform opening (erosion followed by dilation)
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
            # Subtract opened from eroded to get the skeleton
            temp = cv2.subtract(eroded, opened)
            skeleton = cv2.bitwise_or(skeleton, temp)
            # Further erode the image
            eroded = cv2.erode(eroded, kernel)
        return skeleton

    skeleton = skeletonize(canny_edges)

    # Show the results

    cv2.imshow('Skeleton', skeleton)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
