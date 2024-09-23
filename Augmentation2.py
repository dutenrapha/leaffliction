import sys
import os
import shutil
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def aug(origpath):
    img = Image.open(origpath)
    imgfile = os.path.basename(origpath).split('.')[0]
    pathaug = os.path.dirname('images/aug/')

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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Augmentation.py [path_to_image]")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print("Error: The file does not exist")
        sys.exit(1)

    if os.path.exists('images/aug'):
        shutil.rmtree('images/aug')
        print('Augmented images deleted.')
    if not os.path.exists(os.path.join('images/', 'aug')):
        os.makedirs(os.path.join('images/', 'aug'))

    aug(image_path)
