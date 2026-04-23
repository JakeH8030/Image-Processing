# Swan Logo Segmentation & Recognition

A MATLAB/Octave-based computer vision pipeline designed to automatically segment and recognize a specific white swan logo from images with varying backgrounds. This project utilizes robust image processing techniques, avoiding hard-coded thresholds or machine learning.

## 🚀 Key Features

* **Adaptive Segmentation:** Uses Otsu’s Method to calculate optimal thresholds for each individual image based on intensity distribution.
* **HSV Color Analysis:** Processes images in the HSV (Hue, Saturation, Value) color space to isolate the white logo via the Value channel.
* **Connected Component Analysis (CCA):** Automatically identifies the largest object in the scene to isolate the swan from background noise.
* **Performance Evaluation:** Includes a built-in metric to calculate the **Dice Score** against ground truth masks to evaluate accuracy.

## 🛠️ Processing Pipeline

1.  **Preprocessing:** Conversion from RGB to HSV; extraction of the Value (V) channel.
2.  **Binarization:** Application of `graythresh` to determine an adaptive threshold.
3.  **Morphology:** Morphological closing to bridge gaps and `imfill` to ensure a solid silhouette.
4.  **Spatial Filtering:** Labeling regions and retaining only the component with the largest area.

## 📊 Evaluation Metric

The system evaluates success using the **Dice Coefficient**:
$$DS = \frac{2 \cdot |A \cap B|}{|A| + |B|}$$

Where $A$ is the predicted mask and $B$ is the ground truth.

## 📂 Project Structure

* `Task1.m`: Basic segmentation for initial testing (IMG_01.jpg).
* `Task2_3.m`: Full robust pipeline for batch processing and Dice Score evaluation.
* `/images`: Directory for input dataset images (.jpg).
* `/ground_truth`: Directory for annotated mask images (.png).
* `/output`: Generated directory for binary segmentation results.

## ⚙️ Installation & Usage

1.  Clone this repository.
2.  Ensure your images are placed in the `/images` and `/ground_truth` directories.
3.  Run `Task2_3.m` in MATLAB or GNU Octave.
4.  The results will be printed in the console as a table, and binary images will be saved to the `/output` folder.
