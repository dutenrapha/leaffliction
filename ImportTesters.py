import os
import shutil
import requests
import zipfile

ZIP_URL = "https://cdn.intra.42.fr/document/document/17059/test_images.zip"
ZIP_FILE = "test_images.zip"
IMG_FOLDER = "test_images"

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
