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
    
    # Usando cv2.findContours ao invés da função do PlantCV
    contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Criar uma máscara vazia
    kept_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Desenhar todos os contornos na máscara
    cv2.drawContours(kept_mask, contours, -1, (255), -1)
    
    # Salvar a máscara
    cv2.imwrite(os.path.join(savedir, f"{filename}_contours_mask.jpg"), kept_mask)

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
