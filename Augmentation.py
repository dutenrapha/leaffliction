from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import sys
import os
import random

def flip_image(image):
    return ImageOps.mirror(image)

def rotate_image(image):
    angle = random.randint(1,180)
    angle = random.choice([-1,1]) * angle
    return image.rotate(angle)

def skew_image(image):
    xshift = abs(image.size[0] * 0.2)
    new_width = image.size[0] + int(round(xshift))
    return image.transform((new_width, image.size[1]), Image.AFFINE,
                           (1, 0.2, -xshift if xshift > 0 else 0, 0, 1, 0),
                           Image.BICUBIC)

def shear_image(image):
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0.2, 1, 0))

def crop_image(image):
    """
    Crop a random region of the image.

    :param image: PIL Image object.
    :return: Cropped PIL Image object.
    """
    width, height = image.size

    crop_width = 100
    crop_height = 100

    max_x = width - crop_width
    max_y = height - crop_height

    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    right = x + crop_width
    bottom = y + crop_height
    return image.crop((x, y, right, bottom))


def blur_image(image):
    return image.filter(ImageFilter.GaussianBlur(radius=5))

def enhance_contrast(image):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(2.0)  # Increase contrast

def scale_image(image, factor=1.5):
    """
    Scale the image by a given factor.
    
    :param image: PIL Image object.
    :param factor: Scale factor.
    :return: Scaled PIL Image object.
    """
    width, height = image.size
    new_width = int(width * factor)
    new_height = int(height * factor)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def enhance_illumination(image):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(1.5)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Augmentation.py [path_to_image]")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print("Error: The file does not exist")
        sys.exit(1)

    image = Image.open(image_path)
    transformations = {
        'Flip': flip_image,
        'Rotate': rotate_image,
        'Skew': skew_image,
        'Shear': shear_image,
        'Crop': crop_image,
        'Blur': blur_image,
        'Contrast': enhance_contrast,
        'Scaling': scale_image,
        'Illumination': enhance_illumination
    }

    base_name, ext = os.path.splitext(os.path.basename(image_path))

    for name, func in transformations.items():
        temp = image
        transformed_image = func(temp)
        save_path = f"{base_name}_{name}{ext}"
        transformed_image.save(save_path)
        image = temp
        print(f"Saved {save_path}")
