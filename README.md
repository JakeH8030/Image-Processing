# Swan Logo Segmentation System

This project provides an automated image processing pipeline to detect, segment, and recognize a white swan logo from complex backgrounds.

## What It Does

### 1. Automated Segmentation
The system takes raw color images and automatically isolates the swan logo. It specifically:
* **Filters by Intensity:** Uses the HSV color space to focus on brightness, ensuring the white swan is detected even against colorful backgrounds.
* **Self-Adjusting Thresholds:** Uses Otsu’s Method to automatically calculate the best "cut-off" point for every image, meaning no manual brightness settings are required.

### 2. Intelligent Object Recognition
Rather than just picking up every white pixel, the system uses **Connected Component Analysis (CCA)** to understand shapes. It identifies all white objects in an image and intelligently selects only the largest, most contiguous shape—the swan—while discarding background noise and "snow."

### 3. Accuracy Evaluation
The system automatically compares its own results against "Ground Truth" (perfect human-made masks). It calculates a **Dice Score** for each image, which provides a mathematical percentage of how accurately the algorithm's mask matches the real swan.

## Summary of Workflow
1.  **Read Image** -> 2. **HSV Brightness Extraction** -> 3. **Auto-Thresholding** -> 4. **Keep Largest Shape** -> 5. **Fill Internal Holes** -> 6. **Calculate Accuracy Score**.
