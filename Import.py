import os
import shutil
import requests
import zipfile

ZIP_URL = "https://cdn.intra.42.fr/document/document/17060/leaves.zip"
ZIP_FILE = "leaves.zip"
IMG_FOLDER = "images"
SUBFOLDERS = {'apple': 'Apple', 'grape': 'Grape'}

# Remove old folder
if os.path.exists(IMG_FOLDER):
    shutil.rmtree(IMG_FOLDER)
    print("Images deleted.")

# Download and UNZIP
print("Downloading zip file...")
with open(ZIP_FILE, 'wb') as file:
    file.write(requests.get(ZIP_URL).content)
print("Download completed.")

print("Unzipping file...")
with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
    zip_ref.extractall()
os.remove(ZIP_FILE)
print("Zip file deleted and files available.")

# Create folders and move files into subfolders
for subfolder, prefix in SUBFOLDERS.items():
    subfolder_path = os.path.join(IMG_FOLDER, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)
    print(f'Folder "{subfolder_path}" created.')
    for item in os.listdir(IMG_FOLDER):
        item_path = os.path.join(IMG_FOLDER, item)
        if item.startswith(prefix) and os.path.isdir(item_path):
            shutil.move(item_path, subfolder_path)
            print(f'Moved "{item}" into "{subfolder_path}".')
