from huggingface_hub import hf_hub_download
import tarfile
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Download and extract a dataset from Hugging Face Hub.")
parser.add_argument("filename", type=str, help="The name of the .tar.gz file to download and extract.")
args = parser.parse_args()

repo_id = "wifibk/CFNet_Datasets"
filename = args.filename
repo_type = "dataset"
local_dir = "./"

# Download file
file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=local_dir)

print(f"File downloaded to: {file_path}")

# Extract file to the current directory
with tarfile.open(file_path, "r:gz") as tar:
    tar.extractall(path=local_dir)

if filename == "CLCD.tar.gz":
    base_dir = "CLCD"
    os.rename("_CLCD", base_dir)
elif filename == "CLCD-processed.tar.gz":
    base_dir = "CLCD-processed"
    os.rename("CLCD", base_dir)
    
elif filename == "LEVIR-CD.tar.gz":
    base_dir = "LEVIR-CD"
    os.rename("_LEVIR-CD", base_dir)
    
elif filename == "LEVIR-CD-processed.tar.gz":
    base_dir = "LEVIR-CD-processed"
    
elif filename == "SYSU-CD.tar.gz":
    base_dir = "SYSU-CD"

print(f"File extracted to: {local_dir}{base_dir}")

# Delete the downloaded tar.gz file
os.remove(file_path)
print(f"File {file_path} removed")