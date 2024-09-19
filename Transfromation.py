import sys
import cv2
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt

def main(image_path):
    # Inicializa o objeto PlantCV e lê a imagem
    pcv.params.debug = "plot"
    img, path, filename = pcv.readimage(filename=image_path)

    # Aplica um Gaussian Blur
    gaussian_blur = pcv.gaussian_blur(img=img, ksize=(5, 5), sigma_x=0, sigma_y=None)

    # Cria uma máscara binária
    _, thresholded = pcv.threshold.binary(img, 125, 255, 'light')  # Corrigido para 3 argumentos

    # Define uma região de interesse (ROI)
    roi_contour, roi_hierarchy = pcv.roi.rectangle(img=img, x=100, y=100, h=200, w=200)

    # Identifica objetos dentro do ROI
    roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img, 'partial', roi_contour, roi_hierarchy, thresholded)

    # Analisa os objetos
    analysis_image = pcv.analyze_object(img, roi_objects[0], hierarchy)

    # Pseudolandmarks
    _, landmark_image = pcv.pseudolandmarks(img, roi_objects[0], hierarchy)

    # Configura a visualização das imagens em uma grade 2x3
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.ravel()

    axs[0].imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Gaussian Blur')
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(thresholded, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Threshold')
    axs[1].axis('off')

    axs[2].imshow(cv2.cvtColor(kept_mask, cv2.COLOR_BGR2RGB))
    axs[2].set_title('ROI Objects')
    axs[2].axis('off')

    axs[3].imshow(cv2.cvtColor(analysis_image, cv2.COLOR_BGR2RGB))
    axs[3].set_title('Analyze Object')
    axs[3].axis('off')

    axs[4].imshow(cv2.cvtColor(landmark_image, cv2.COLOR_BGR2RGB))
    axs[4].set_title('Pseudolandmarks')
    axs[4].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Transformation.py [path_to_image]")
        sys.exit(1)
    main(sys.argv[1])
