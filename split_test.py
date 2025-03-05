import random
import shutil
from pathlib import Path


def split_test_dataset(source_dir: str, test_dir: str, percentage: float):
    """
    Splits a percentage of images from subdirectories in a source
    directory into a test directory, preserving the folder structure.

    :param source_dir: Path to the source directory containing
    subdirectories of images.
    :param test_dir: Path to the target directory for test images.
    :param percentage: Percentage of images to be moved to
    the test directory (0 to 1).
    """
    source_dir = Path(source_dir)
    test_dir = Path(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    for subdir in source_dir.iterdir():
        if subdir.is_dir():
            images = list(subdir.glob('*'))
            num_images = len(images)
            num_test_images = int(num_images * percentage)

            test_images = random.sample(images, num_test_images)

            test_subdir = test_dir / subdir.name
            test_subdir.mkdir(parents=True, exist_ok=True)

            for img in test_images:
                shutil.move(str(img), test_subdir / img.name)

    print(f"Split completed. Test images moved to '{test_dir}'.")


split_test_dataset("images/grape", "test_dataset_grape", 0.2)
split_test_dataset("images/apple", "test_dataset_apple", 0.2)
