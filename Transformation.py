import sys
import os
import shutil
import cv2
import numpy as np
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt

def is_jpg(filename):
    return filename.lower().endswith('.jpg')

def generate_output_dir(full_path, output_dir):
    """
    Substitute the root directory of a full file path with a specified output directory, excluding the filename.

    Parameters:
    - full_path (str): The full path of the file.
    - output_dir (str): The new root directory.

    Returns:
    - str: The modified path with the new root directory, excluding the filename.
    """
    # Split the full path into components
    path_components = full_path.split(os.sep)
    
    # Remove the first component (the original root) and the last component (the filename)
    relative_components = path_components[1:-1]
    
    # Construct the new path with the output directory as the root
    new_path = os.path.join(output_dir, *relative_components)
    
    return new_path

def transform(origpath, output_path, show_img):
    img = cv2.imread(origpath)
    filename = os.path.basename(origpath).split('.')[0]
    
    #savedir = os.path.dirname(output_path)
    savedir = generate_output_dir(origpath, output_path)

    #print(savedir, filename)
    blur_gaussiano = cv2.GaussianBlur(img, (15, 15), 0)
    cv2.imwrite(os.path.join(savedir, f"{filename}_gaussian_blur.jpg"), blur_gaussiano)

    blur_gray = cv2.cvtColor(blur_gaussiano, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(savedir, f"{filename}_gaussian_blur_bw.jpg"), blur_gray)

    _, mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(savedir, f"{filename}_mask.jpg"), mask)

    pcv.params.debug = None
    s = pcv.rgb2gray_hsv(rgb_img=img, channel="s")
    thresholded = pcv.threshold.binary(gray_img=s, threshold=125, object_type='light')
    
    # Usando cv2.findContours ao invés da função do PlantCV
    contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Criar uma máscara vazia
    kept_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Desenhar todos os contornos na máscara
    cv2.drawContours(kept_mask, contours, -1, (255), -1)
    
    # Salvar a máscara
    cv2.imwrite(os.path.join(savedir, f"{filename}_contours_mask.jpg"), kept_mask)


    # Laplace Filter
    lp_img = pcv.laplace_filter(gray_img=img, ksize=3, scale=1)
    cv2.imwrite(os.path.join(savedir, f"{filename}_laplace.jpg"), lp_img)

    # Sobel filter
    sobel = pcv.sobel_filter(gray_img=img, dx=1, dy=0, ksize=3)
    cv2.imwrite(os.path.join(savedir, f"{filename}_sobel.jpg"), sobel)

    # Display images if necessary
    if show_img:
        cols = 7
        rows = 1
        fig, axs = plt.subplots(rows, cols, figsize=(20, 5))
        axs = axs.ravel()

        axs[0].imshow(img)
        axs[0].set_title('Original')
        axs[0].axis('off')
        axs[1].imshow(blur_gaussiano)
        axs[1].set_title('Blur Gaussiano')
        axs[1].axis('off')
        axs[2].imshow(blur_gray)
        axs[2].set_title('Blur BW')
        axs[2].axis('off')
        axs[3].imshow(mask)
        axs[3].set_title('Contour Mask')
        axs[3].axis('off')
        axs[4].imshow(kept_mask)
        axs[4].set_title('Mask')
        axs[4].axis('off')
        axs[5].imshow(lp_img)
        axs[5].set_title('Laplace Filter')
        axs[5].axis('off')
        axs[6].imshow(sobel)
        axs[6].set_title('Sobel Filter')
        axs[6].axis('off')

        plt.tight_layout()
        plt.show()

        print(f"Imagens transformadas salvas em {savedir}")

def delete_transformed_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if 'mask' in file or 'blur' in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)

def process_directory(directory, output_path):
    
    new_dir = directory
    #delete_transformed_files(new_dir)

    # Process subdirectories
    for root, _, files in os.walk(new_dir):
        print("Processing: ", root)
        #orig_files = [item for item in files if '_' not in item]
        orig_files = files
        for file in orig_files:
            if is_jpg(file):
                file_path = os.path.join(root, file)
                transform(file_path, output_path, False)

if __name__ == "__main__":
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python Transformation.py [path_to_image]: process single file and display images")
        print("Usage: python Transformation.py [path_to_dir] [output_path]: process all files in path_to_dir and saves in output_path (must be different)")
        sys.exit(1)

    # Process single file and display images
    if len(sys.argv) == 2:
        cli_arg = sys.argv[1]
        if os.path.isfile(cli_arg):
            transform(cli_arg, cli_arg, True)
        else:
            print(f"Invalid argument: {cli_arg}")
    # Process all JPG files in subdirs
    if len(sys.argv) == 3:
        cli_arg = sys.argv[1]
        output_path = sys.argv[2]
        # Create new directory if necessary
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        shutil.copytree(cli_arg, output_path)
        if os.path.isdir(cli_arg) and os.path.isdir(output_path):
            process_directory(cli_arg, output_path)
        else:
            print(f"Invalid argument: {cli_arg}")
