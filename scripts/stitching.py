import cv2
import numpy as np
import matplotlib.pyplot as plt

# Loading the images to bew stitched
img1 = cv2.imread("../input_images/room2.jpg")
img2 = cv2.imread("../input_images/room1.jpg")

# Converting the images from BGR to RGB
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


# Converting the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initializing SIFT detector
sift = cv2.SIFT_create()

# Detecting the keypoints and compute descriptors
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Drawing the keypoints on both images for visulization
img1_with_keypoints = cv2.drawKeypoints(img1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_with_keypoints = cv2.drawKeypoints(img2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# Displaying the keypoints detected in both images.
plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
plt.title("Keypoints in img1")
plt.imshow(img2_with_keypoints)

plt.subplot(1,2,2)
plt.title("Keypoints in img2")
plt.imshow(img1_with_keypoints)

# Creating a BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Matching the descriptors from both the images 
matches = bf.match(descriptors1, descriptors2)

# Sorting the matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the top 10 best matches for visualization
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Displaying the matched keypoints
plt.figure(figsize=(12, 6))
plt.imshow(img_matches)
plt.title("Image Matches")

# Selecting the top 50 matches for homography estimation
good_matches = matches[:50]

# Extracting matched keypoints cordinates
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Computing homography matrix using RANSAC
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Getting image dimensions
height, width, channels = img2.shape  # Ensure color image processing

# Warping the img1 using the homography matrix onto img2 perspective
warped_img1 = cv2.warpPerspective(img1, H, (width * 2, height))

# Overlay img2 onto the warped image
warped_img1[0:height, 0:width] = img2  # Ensuring correct placement

# This function is used to remove the black region that forms after the stitching process
def crop_black_region(image):
    # Converting to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Finding all non-black pixels
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Getting the bounding box of non-black regions
    coords = cv2.findNonZero(thresh)  # Find non-black pixels
    x, y, w, h = cv2.boundingRect(coords)  # Get bounding box

    # Croping the image
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

# Applying cropping after stitching
stitched_image = crop_black_region(warped_img1)


# Displaying the final panorama
plt.figure(figsize=(12, 6))
plt.imshow(stitched_image)
plt.title("Final Panorama")
plt.show()


#TVS Chaitanya
#IMT2022545



