from fastai.data.external import untar_data, URLs
import shutil
import os

# Step 1: Download MNIST to the default fastai location
default_path = untar_data(URLs.MNIST)

# Step 2: Define desired target directory
target_path = "."

# Step 3: Move the downloaded files to the target directory
# Ensure the target directory exists
os.makedirs(target_path, exist_ok=True)
# Move the extracted MNIST dataset to the target directory
shutil.move(default_path, target_path)

os.rename(default_path.name, "data")
print(f"Dataset downloaded to ./data")