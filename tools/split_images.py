import os
import argparse
from PIL import Image

# Define a function to handle image splitting
def split_images(input_folder, output_folder):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Only process image files
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            
            # Check if the image size is 512x512
            if img.size == (512, 512):
                base_name = os.path.splitext(filename)[0]  # Get the file name without extension
                
                # Split the image into 4 256x256 pieces
                for i in range(2):
                    for j in range(2):
                        left = 256 * j
                        upper = 256 * i
                        right = left + 256
                        lower = upper + 256
                        cropped_img = img.crop((left, upper, right, lower))
                        
                        # Generate a new file name and save the cropped image
                        new_filename = f"{base_name}_{i * 2 + j}.png"
                        output_path = os.path.join(output_folder, new_filename)
                        cropped_img.save(output_path)
                        
                print(f"Processed: {filename}")
            else:
                print(f"Skipped: {filename} (Size is not 512x512)")

    print("All images processed.")

# Set up command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split 512x512 images into 4 256x256 images")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing 512x512 images")
    parser.add_argument("output_folder", type=str, help="Path to the output folder where 256x256 images will be saved")

    args = parser.parse_args()

    # Call the function to process images
    split_images(args.input_folder, args.output_folder)

