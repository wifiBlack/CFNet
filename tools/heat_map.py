import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from configs import *
from dataset.dataset import MyDataset
from confuse_matrix import *
from model.CFNet import CFNet
import torch.nn as nn
from torch.amp import autocast

import matplotlib.pyplot as plt
import os


def save_heatmap(tensor, output_dir, base_name, batch_index, i):
    """
    Saves a heatmap image for the given tensor.

    Args:
        tensor (torch.Tensor): The tensor to save as a heatmap.
        output_dir (str): The directory to save the heatmap image to.
        base_name (str): The base name of the image file.
        batch_index (int): The index of the current batch.
        i (int): The index of the current sample in the batch.
    """
    data = tensor.cpu().numpy()

    # Get min and max values from the data to ensure correct color mapping
    vmin, vmax = data.min(), data.max()

    # Plot the heatmap without the color bar, using dynamic vmin and vmax
    plt.imshow(data, cmap='coolwarm', vmin=vmin, vmax=vmax)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct the output file path
    output_path = os.path.join(output_dir, f"{base_name}_sample_{batch_index}_RM_{i}.png")

    # Save the heatmap to the output file (without the color bar)
    plt.axis('off')  # Turn off axes
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to free up memory


def save_pixel_distribution(tensor, output_dir, base_name, batch_index, i):
    """
    Saves a histogram of pixel value distribution for the given tensor.

    Args:
        tensor (torch.Tensor): The tensor to analyze.
        output_dir (str): The directory to save the histogram image to.
        base_name (str): The base name of the image file.
        batch_index (int): The index of the current batch.
        i (int): The index of the current sample in the batch.
    """
    data = tensor.cpu().numpy().flatten()  # Flatten the tensor to 1D for histogram
    
    # Plot the histogram
    plt.figure()
    plt.hist(data, bins=50, color='blue', alpha=0.7)  # 50 bins for better granularity
    plt.title("Pixel Value Distribution")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct the output file path
    output_path = os.path.join(output_dir, f"{base_name}_sample_{batch_index}_RM_{i}_distribution.png")

    # Save the histogram to the output file
    plt.savefig(output_path)
    plt.close()  # Close the plot to free up memory


def heat_map():

    args = parse_arguments()
    data_path = args.data_dir
    gpus = str(args.gpu[0])[2]
    if len(args.gpu) > 1:
        for gpu in args.gpu[1:]:
            gpus = gpus + "," + str(gpu)[2]
    batch_size = args.batch_size
    num_workers = args.num_workers
    checkpoint = args.checkpoint
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    test_data = MyDataset(data_path, "test")
    print("Get heatmaps on {} image-pairs".format(len(test_data)))
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CFNet(3, 3)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    if checkpoint != None:
        model.load_state_dict(torch.load(checkpoint, weights_only=False))
        print("Checkpoint {} succesfully loaded".format(checkpoint))
        
    model = model.to(device)
    model.eval()

    output_dir = 'output_heatmaps'
    loop = tqdm(test_loader, position=0, leave=False)
    with torch.no_grad():
        for x1, x2, target, file_name in loop:
            x1 = x1.to(device).float()
            x2 = x2.to(device).float()
            target = target.to(device).float()
            with autocast(device.type+":"+str(device.index)):                
                RM = model(x1, x2, device, is_RM=True)
            
            for batch_index in range(RM[0].shape[0]): 
                for i, rm_tensor in enumerate(RM):
                    save_heatmap(rm_tensor[batch_index], output_dir, file_name[batch_index], batch_index, i)
                    save_pixel_distribution(rm_tensor[batch_index], output_dir, file_name[batch_index], batch_index, i)

    
if __name__ == "__main__":
    heat_map()
