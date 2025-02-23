import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, color

def detect_coins(image_path):
    # Loading the image of the coins
    img = cv2.imread(image_path)

    # Creating a copy to draw tnhe contour lines
    detected_img = img.copy()

    # Converting BGR to RGB
    detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Converting to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Applying GaussianBlur to smoothen the image ans reduce noise.
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 2)

    # Applying thresholding to create a binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Here we are trying to remove the noise in the image using morphological operatinos
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Here we are identifying connected components to label different regions
    labels = measure.label(cleaned, connectivity=2)
    colored_labels = color.label2rgb(labels, bg_label=0)

    # Finding the contours of the detected image
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Drawing the detected contours
    cv2.drawContours(detected_img, contours, -1, (255, 0, 0), 5)


    # Displaying the original and image with detected coins side by side
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(original_image)

    plt.subplot(1,2,2)
    plt.title("Detected Coins")
    plt.imshow(detected_img)

    # Display segmented coins
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)

    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(colored_labels)
    plt.show()

    return len(contours)
    


image_path = "../input_images/coins.jpg"

num_coins=detect_coins(image_path)
print(f"Detected {num_coins} coins")
plt.show()

#IMT2022545
#TVS Chaitanya