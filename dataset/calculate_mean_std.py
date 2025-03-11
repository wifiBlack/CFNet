import cv2
import numpy as np
import os

def calculate_mean_std(image_path):
    """Calculates the mean and standard deviation of an image

    Args:
        image_path: Path to the image

    Returns:
        tuple: A tuple containing the mean and standard deviation (both 3-element vectors)
    """

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean, std = cv2.meanStdDev(img.astype(np.float64))
    return mean, std

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Calculate mean and std of images in a folder.')
    parser.add_argument('folder_path', type=str, help='Path to the image folder')
    args = parser.parse_args()

    total_mean = np.zeros([3,])  # Ensure correct initialization
    total_std = np.zeros([3,])
    image_count = 0

    for filename in os.listdir(args.folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(args.folder_path, filename)
            mean, std = calculate_mean_std(image_path)
            total_mean += mean[:,0]
            total_std += std[:,0]
            image_count += 1

    mean = total_mean / image_count
    std = total_std / image_count

    print("mean:", mean / 255.0)
    print("std:", std / 255.0)