import sys
import os
import shutil
import cv2
import numpy as np
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt

def is_jpg(filename):
    return filename.lower().endswith('.jpg')

def transform(origpath, show_img):
    img = cv2.imread(origpath)
    filename = os.path.basename(origpath).split('.')[0]
    
    savedir = os.path.dirname(origpath)

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

    # Display images if necessary
    if show_img:
        cols = 5
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

        plt.tight_layout()
        plt.show()

        print(f"Imagens transformadas salvas em {savedir}")

def delete_transformed_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if 'mask' in file or 'blur' in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)

def process_directory(directory):
    new_dir = directory
    delete_transformed_files(new_dir)
    
    # Process subdirectories
    for root, _, files in os.walk(new_dir):
        print("Processing: ", root)
        #orig_files = [item for item in files if '_' not in item]
        orig_files = files
        for file in orig_files:
            if is_jpg(file):
                file_path = os.path.join(root, file)
                transform(file_path, False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Transformation.py [path_to_image]: process single file and display images")
        print("Usage: python Transformation.py [path_to_dir]: process all files in the directory")
        sys.exit(1)

    cli_arg = sys.argv[1]
    if os.path.isfile(cli_arg):
        # Process single file and display images
        transform(cli_arg, True)
    elif os.path.isdir(cli_arg):
        # Process all JPG files in subdirs
        process_directory(cli_arg)
    else:
        print(f"Invalid argument: {cli_arg}")