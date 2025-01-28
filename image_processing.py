import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# Load the image
import re
from skimage.morphology import skeletonize
from skimage import img_as_bool


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


father_path = '/home/jian/Downloads/results-seed-1357-002/results-ddim-sampling_steps-20-seed-1357/epoch-999/images'
image_lists = sorted(os.listdir(father_path), key=natural_sort_key)


def count_fingers(image_path):

    # Step 1: Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image!")
        return
    image = cv2.resize(image, (600, 600))  # Resize for consistency
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Threshold the image
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 3: Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No hand detected!")
        return

    # Step 4: Find the largest contour (assuming it's the hand)
    max_contour = max(contours, key=cv2.contourArea)

    # Step 5: Convex Hull and Convexity Defects
    hull = cv2.convexHull(max_contour)
    hull_indices = cv2.convexHull(max_contour, returnPoints=False)
    defects = cv2.convexityDefects(max_contour, hull_indices)

    finger_count = 0

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]  # start, end, far, depth
            start = tuple(max_contour[s][0])
            end = tuple(max_contour[e][0])
            far = tuple(max_contour[f][0])
            print(f"Depth: {d}")

            # Use triangle geometry to detect fingers
            a = np.linalg.norm(np.array(start) - np.array(end))
            b = np.linalg.norm(np.array(start) - np.array(far))
            c = np.linalg.norm(np.array(end) - np.array(far))
            angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))  # Cosine rule

            # Angle should be less than 90 degrees to detect fingers
            # if angle < np.pi / 2 and d > 10000:  # Threshold for depth
            finger_count += 1
            cv2.circle(image, far, 5, (0, 255, 0), -1)

    # Draw the hand contour and the convex hull
    cv2.drawContours(image, [max_contour], -1, (255, 0, 0), 2)  # Hand contour in blue
    cv2.drawContours(image, [hull], -1, (0, 255, 255), 2)  # Convex hull in yellow

    # Display the finger count
    cv2.putText(image, f"Fingers: {finger_count + 1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the image
    cv2.imshow("Hand with Hull", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


for image_name in image_lists:
    image_path = os.path.join(father_path, image_name)
    finger_count = count_fingers(image_path)
    print(f"Number of fingers detected: {finger_count}")
    # kernel = np.ones((3, 3), np.uint8)
    # canny_edges = cv2.dilate(canny_edges, kernel, iterations=1)

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

    # skeleton = np.array(skeletonize(img_as_bool(image)), dtype=np.uint8) * 255

    # Show the results

    # cv2.imshow('Skeleton', skeleton)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
