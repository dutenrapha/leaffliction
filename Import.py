import os
import shutil
import requests
import zipfile

if os.path.exists('images'):
    shutil.rmtree('images')
    print('Images deleted.')

zipfileurl = "leaves.zip"
print("Downloading zip file...")
rq = requests.get("https://cdn.intra.42.fr/document/document/17060/leaves.zip")
with open(zipfileurl, 'wb') as arquivo:
    arquivo.write(rq.content)
print("Download completed.")

print("Unzip file...")
with zipfile.ZipFile(zipfileurl, 'r') as zip_ref:
    zip_ref.extractall(os.getcwd())
print("Files available.")

os.remove(zipfileurl)
print("Zip file deleted.")

imgfld = 'images/'
if not os.path.exists(os.path.join(imgfld, 'apple')):
    os.makedirs(os.path.join(imgfld, 'apple'))
    print('Pasta "apple" criada dentro de "images".')

if not os.path.exists(os.path.join(imgfld, 'grape')):
    os.makedirs(os.path.join(imgfld, 'grape'))
    print('Pasta "grape" criada dentro de "images".')

for item in os.listdir(imgfld):
    item_path = os.path.join(imgfld, item)
    if item.startswith('Apple') and os.path.isdir(item_path):
        shutil.move(item_path, os.path.join(imgfld, 'apple'))
        print(f'Movendo pasta {item} para a pasta "apple" dentro de "images".')

for item in os.listdir(imgfld):
    item_path = os.path.join(imgfld, item)
    if item.startswith('Grape') and os.path.isdir(item_path):
        shutil.move(item_path, os.path.join(imgfld, 'grape'))
        print(f'Movendo pasta {item} para a pasta "grape" dentro de "images".')
