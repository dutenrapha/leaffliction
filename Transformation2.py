import sys
import os
import shutil
import cv2
import numpy as np
from plantcv import plantcv as pcv

def transform(origpath):
    img = cv2.imread(origpath)
    filename = os.path.basename(origpath).split('.')[0]
    
    savedir = 'images/transform'

    blur_gaussiano = cv2.GaussianBlur(img, (15, 15), 0)
    cv2.imwrite(os.path.join(savedir, f"{filename}_gaussian_blur.jpg"), blur_gaussiano)

    blur_gray = cv2.cvtColor(blur_gaussiano, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(savedir, f"{filename}_gaussian_blur_bw.jpg"), blur_gray)

    _, mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(savedir, f"{filename}_mask.jpg"), mask)

    pcv.params.debug = None
    s = pcv.rgb2gray_hsv(rgb_img=img, channel="s")
    thresholded = pcv.threshold.binary(gray_img=s, threshold=125, object_type='light')
    
    # Definir uma região de interesse (ROI)
    contours, hierarchy = pcv.find_objects(img=img, mask=thresholded)
    
    # Definir uma região de interesse (ROI)
    roi_contour, roi_hierarchy = pcv.roi.rectangle(img=img, x=0, y=0, h=256, w=256)
    
    # Agrupar objetos dentro da ROI
    roi_objects, hierarchy3, kept_mask, obj_area = pcv.cluster_contours(
        img=img, roi_type='partial', roi_contour=roi_contour, object_hierarchy=hierarchy, object_contour=contours
    )
    
    # Salvar ROI objects
    cv2.imwrite(os.path.join(save_dir, f"{filename}_roi_objects.jpg"), kept_mask)



    print(f"Imagens transformadas salvas em {savedir}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Augmentation.py [path_to_image]")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print("Error: The file does not exist")
        sys.exit(1)
    
    if os.path.exists('images/transform'):
        shutil.rmtree('images/transform')
        print('Transformed images deleted.')
    if not os.path.exists(os.path.join('images/', 'transform')):
        os.makedirs(os.path.join('images/', 'transform'))

    transform(sys.argv[1])
