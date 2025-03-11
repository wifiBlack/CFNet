import os
import shutil
import cv2
import numpy as np
import argparse

def process_images(prediction_folder, ground_truth_folder, output_folder):
    """
    Compare prediction and ground truth images and save the result.

    Args:
        prediction_folder (str): Path to the prediction folder.
        ground_truth_folder (str): Path to the ground truth folder.
        output_folder (str): Path to save the output images.
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        print(f"Deleted existing directory: {output_folder}")
        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_names = os.listdir(prediction_folder)

    for file_name in file_names:
        pred_path = os.path.join(prediction_folder, file_name)
        gt_path = os.path.join(ground_truth_folder, file_name)
        
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_img = (gt_img / 255).astype(np.uint8)
        
        if pred_img.shape != gt_img.shape:
            print(f"Image size mismatch: {file_name}")
            continue

        result_img = np.zeros((gt_img.shape[0], gt_img.shape[1], 3), dtype=np.uint8)

        # Red: Predicted as positive, but ground truth is negative
        result_img[(gt_img == 1) & (pred_img == 0)] = [0, 0, 255]  

        # Green: Predicted as negative, but ground truth is positive
        result_img[(gt_img == 0) & (pred_img == 1)] = [0, 255, 0]  

        # White: Predicted as positive, and ground truth is positive
        result_img[(gt_img == 1) & (pred_img == 1)] = [255, 255, 255]  

        # Black: Predicted as negative, and ground truth is negative
        result_img[(gt_img == 0) & (pred_img == 0)] = [0, 0, 0]  

        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, result_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare prediction and ground truth images and save the result.")
    parser.add_argument('--prediction_folder', type=str, required=True, help='Path to the prediction folder')
    parser.add_argument('--ground_truth_folder', type=str, required=True, help='Path to the ground truth folder')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to save the output images')

    args = parser.parse_args()

    process_images(args.prediction_folder, args.ground_truth_folder, args.output_folder)
