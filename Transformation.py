import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_image_transformations(image_path, output_dir):
    """Applies various image transformations and saves the results.

    These transformations are designed to extract key features from
    plant images, which can be useful for tasks like plant
    identification or disease detection.

    Args:
        image_path (str): Path to the original image.
        output_dir (str): Path to the directory where the transformed
            images will be saved.
    """
    img = cv2.imread(image_path)  # Loads the image using OpenCV
    filename, ext = os.path.splitext(os.path.basename(image_path))
    # Extracts filename and extension

    # 1. Gaussian Blur
    # Reduces noise and fine details by averaging pixel values.
    # Contributes to feature extraction by:
    #   - Simplifying the image, making it easier to identify
    #     broader shapes.
    #   - Reducing the impact of minor imperfections or noise on
    #     subsequent analysis.
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_GaussianBlur{ext}"),
                blurred_img)

    # 2. Mask (using color segmentation)
    # Isolates regions based on color range (assumes green leaves).
    # Contributes to feature extraction by:
    #   - Segmenting the leaf from the background, allowing focus on
    #     leaf-specific features.
    #   - Creating a binary image where the leaf is white and the rest
    #     is black, simplifying analysis.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Convert to HSV color space (easier for color-based segmentation)
    lower_green = np.array([35, 40, 40])  # Lower bound for green color range
    upper_green = np.array([85, 255, 255])  # Upper bound for green color range
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Create a mask based on the color range
    cv2.imwrite(os.path.join(output_dir, f"{filename}_Mask{ext}"), mask)

    # 3. ROI Objects (Region of Interest - Contours)
    # Identifies outlines of objects in the binary mask.
    # Contributes to feature extraction by:
    #   - Enabling shape-based analysis, such as leaf area, perimeter,
    #     and shape descriptors.
    #   - Providing information about the number of leaves in the image.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # Find contours in the mask
    img_contours = img.copy()
    # Create a copy of the original image to draw contours on
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    # Draws contours in green
    cv2.imwrite(os.path.join(output_dir, f"{filename}_ROIObjects{ext}"),
                img_contours)

    # 4. Analyze Object (Largest Contour)
    # Selects the largest contour, assuming it's the main leaf.
    # Contributes to feature extraction by:
    #   - Focusing on the primary leaf, reducing the impact of noise
    #     or smaller objects.
    #   - Simplifying the analysis by considering only the most
    #     relevant object in the image.
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Find the contour with the largest area
        img_largest_contour = img.copy()
        # Create a copy of the original image
        cv2.drawContours(img_largest_contour, [largest_contour], -1,
                         (255, 0, 255), 3)
        # Draws the largest contour in magenta
        cv2.imwrite(os.path.join(output_dir,
                                 f"{filename}_AnalyzeObject{ext}"),
                    img_largest_contour)

    # 5. Pseudolandmarks (Extremal Points of Contour)
    # Finds the leftmost, rightmost, topmost, and bottommost points of
    # the contour.
    # Contributes to feature extraction by:
    #   - Providing key reference points for measuring leaf shape and
    #     orientation.
    #   - Enabling the calculation of shape-based features such as leaf
    #     length, width, and aspect ratio.
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Find the contour with the largest area
        extLeft = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
        # Find the leftmost point
        extRight = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
        # Find the rightmost point
        extTop = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
        # Find the topmost point
        extBot = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
        # Find the bottommost point

        img_landmarks = img.copy()  # Create a copy of the original image
        cv2.circle(img_landmarks, extLeft, 8, (0, 0, 255), -1)  # Red
        cv2.circle(img_landmarks, extRight, 8, (0, 255, 255), -1)  # Yellow
        cv2.circle(img_landmarks, extTop, 8, (255, 0, 0), -1)  # Blue
        cv2.circle(img_landmarks, extBot, 8, (255, 255, 0), -1)  # Cyan

        cv2.imwrite(os.path.join(output_dir,
                                 f"{filename}_Pseudolandmarks{ext}"),
                    img_landmarks)

    # 6. Color Histogram
    # Plots and saves the color histogram of the image.
    # Contributes to feature extraction by:
    #   - Providing information about the color distribution in the leaf.
    #   - Enabling the detection of color changes or abnormalities
    #     that may indicate disease.
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # Calculate the color histogram

    # Plot the histogram
    plt.figure()
    plt.title("Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)  # Plot the histogram
    plt.xlim([0, 256])  # Set the x-axis limits

    # Save the plot as a JPG
    plt.savefig(os.path.join(output_dir, f"{filename}_ColorHistogram.jpg"))
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        output_dir = os.path.dirname(image_path)
        output_dir = "./"
        apply_image_transformations(image_path, output_dir)
    else:
        print("Usage: python transformation.py <image_path>")
