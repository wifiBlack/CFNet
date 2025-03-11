import os
from PIL import Image
import sys

def split_image_with_overlap(image_path, output_folder):
    """Split a 1024x1024 image into 25 256x256 images with 64 pixels overlap and save them to the output folder"""
    img = Image.open(image_path)
    img_width, img_height = img.size

    if img_width != 1024 or img_height != 1024:
        print(f"Skipping image {image_path} because it is not 1024x1024 in size")
        return
    
    base_name = os.path.basename(image_path).split('.')[0]  # Get the base name of the image (without extension)

    step_size = 192  # Step size for each window
    patch_size = 256  # Size of each sub-image
    count = 0

    # Split the image with a step size of 192, generating 25 sub-images
    for i in range(0, img_width - patch_size + 1, step_size):
        for j in range(0, img_height - patch_size + 1, step_size):
            # Define the area to be cropped
            box = (i, j, i + patch_size, j + patch_size)
            part = img.crop(box)
            
            # Save the cropped image
            output_path = os.path.join(output_folder, f"{base_name}_{count}.png")
            part.save(output_path)
            count += 1

    print(f"Image {image_path} split completed and saved to {output_folder}")

def process_folder(input_folder, output_folder):
    """Process all 1024x1024 PNG images in the input folder"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(input_folder, filename)
            split_image_with_overlap(image_path, output_folder)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split_images_with_overlap.py <input folder path> <output folder path>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    process_folder(input_folder, output_folder)
