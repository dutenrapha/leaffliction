import sys
import os
import shutil
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from collections import defaultdict

def is_jpg(filename):
    return filename.lower().endswith('.jpg')

def aug(origpath, show_img):
    img = Image.open(origpath)
    imgfile = os.path.basename(origpath).split('.')[0]
    pathaug = os.path.dirname(origpath)

    augimages = {
        'Flip': ImageOps.mirror(img),
        'Rotate': img.rotate(90),
        'Skew': img.transpose(Image.Transpose.ROTATE_180),
        'Shear': img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        'Crop': img.crop((40, 40, img.size[0] - 40, img.size[1] - 40)),
        'Distortion': img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    }

    for nome, img_transformada in augimages.items():
        img_transformada.save(f"{pathaug}/{imgfile}_{nome}.jpg")

    if show_img:
        cols = 7
        rows = 1
        fig, axs = plt.subplots(rows, cols, figsize=(20, 5))
        axs = axs.ravel()

        axs[0].imshow(img)
        axs[0].set_title('Original')
        axs[0].axis('off')
        for i, (name, img) in enumerate(augimages.items()):
            axs[i+1].imshow(img)
            axs[i+1].set_title(name)
            axs[i+1].axis('off')

        for j in range(i+1, rows*cols):
            axs[j].axis('off')

        plt.tight_layout()
        plt.show()

def process_directory(directory):
    # Create new directory with 'augmented' prefix
    new_dir = f"augmented_{os.path.basename(directory)}"
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    shutil.copytree(directory, new_dir)
    
    # Find subdirectory with most files
    subdir_file_counts = defaultdict(int)
    for root, _, files in os.walk(new_dir):
        subdir_file_counts[root] = len(files)
    
    max_files_subdir = max(subdir_file_counts, key=subdir_file_counts.get)
    n = subdir_file_counts[max_files_subdir]
    
    # Process subdirectories
    for root, _, files in os.walk(new_dir):
        print("Processing: ", root)
        if root != max_files_subdir:
            for file in files:
                if is_jpg(file):
                    file_path = os.path.join(root, file)
                    aug(file_path, False)
            
            # Delete newest files if count exceeds n
            files = [os.path.join(root, f) for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            while len(files) > n:
                os.remove(files.pop(0))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Augmentation.py [path_to_image]: process single file and display images")
        print("Usage: python Augmentation.py [path_to_dir]: process all files in the directory")
        sys.exit(1)

    cli_arg = sys.argv[1]
    if os.path.isfile(cli_arg):
        # Process single file and display images
        aug(cli_arg, True)
    elif os.path.isdir(cli_arg):
        # Copy 'dir' to 'augmented_dir' and process all files, without displaying images
        process_directory(cli_arg)
    else:
        print(f"Invalid argument: {cli_arg}")