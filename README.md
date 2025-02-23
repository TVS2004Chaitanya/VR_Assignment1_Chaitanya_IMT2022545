# Visual Recognition Assignment-1

## Author

**TVS Chaitanya**\
**IMT2022545**

## Overview

This repository contains two Python scripts for **coin detection** and **image stitching** using OpenCV.

### Files in This Repository

1. **coin\_detection.py** – Detects, counts, and segments coins in an image.
2. **stitching.py** – Stitches two images together to create a panorama.
3. **requirements.txt** – Lists the dependencies required for running the scripts.

### Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

---

## Part 1: Coin Detection

### Running the Script

Run the following command to execute the coin detection script:

```sh
python coin_detection.py
```

### Functionality

- Converts the image to **grayscale** and applies **Gaussian blur** to reduce noise.
- Uses **Otsu’s thresholding** and **morphological opening** to segment the coins.
- Extracts contours using `cv2.findContours()` and highlights detected coins.
- Uses `color.label2rgb()` to colorize segmented regions and prints the total coin count.

### Key Functions Used

- `cv2.GaussianBlur()` – Applies Gaussian filter for noise reduction.
- `cv2.threshold()` – Performs image thresholding.
- `cv2.morphologyEx()` – Executes morphological transformations.
- `skimage.measure.label()` – Assigns unique labels to segmented objects.
- `skimage.color.label2rgb()` – Assigns colors to labeled regions for visualization.

---

## Part 2: Image Stitching

### Running the Script

Run the following command to execute the image stitching script:

```sh
python stitching.py
```

### Functionality

- Detects keypoints using **SIFT** and matches them using **BFMatcher**.
- Computes a **homography matrix** using **RANSAC** for image alignment.
- Warps the first image to match the second, then blends them together.
- Crops out black regions for a cleaner final panorama.

### Key Functions Used

- `cv2.SIFT_create()` – Initializes the SIFT detector.
- `sift.detectAndCompute()` – Detects keypoints and computes descriptors.
- `cv2.drawKeypoints()` – Visualizes detected keypoints.
- `cv2.BFMatcher()` – Matches keypoints using a brute-force approach.
- `bf.match()` – Finds the best matches between feature descriptors.
- `cv2.drawMatches()` – Draws lines connecting matching keypoints.
- `cv2.findHomography()` – Computes the homography matrix for image transformation.
- `crop_black_region()` – Crops black regions after warping.

---

## Output

- **Part 1**: Detected coins with segmentation and total count displayed.  
***Detected Coins***
  ![detected_coins](output_images/detected_coins.png)  
***Segmented Coins***  
  ![detected_coins](output_images/segmented_coins.png)  
- **Part 2**: Final stitched panorama from the input images.  
  ***Detected KeyPoints***  
  ![detected_coins](output_images/Keypoints.png)
  ***Image Matches***  
  ![detected_coins](output_images/Image_Matches.png)
  ***Final Panorama***
  ![detected_coins](output_images/Final_Panorama.png)
  

---


