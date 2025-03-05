import sys
import os
import shutil
from PIL import Image, ImageOps


def is_jpg(filename):
    return filename.lower().endswith('.jpg')


def aug(origpath, filedir):
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

    for nome, img_transfor in augimages.items():
        if filedir == "dir":
            img_transfor.save(f"{pathaug}/{imgfile}_{nome}.jpg")
        else:
            img_transfor.save(f"{imgfile}_{nome}.jpg")


def process_directory(directory, dest_dir="augmented"):
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    total_files = sum(len(files) for _, _, files in os.walk(directory))
    print(f"Total Files: {total_files}")

    counter = 0
    for root, _, files in os.walk(directory):
        dest_path = os.path.join(dest_dir, os.path.relpath(root, directory))
        os.makedirs(dest_path, exist_ok=True)

        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_path, file)
            shutil.copy(src_file, dest_file)
            counter += 1
            sys.stdout.write(f"\rFiles completed: {counter}/{total_files}")
            sys.stdout.flush()
    print("\n")

    min_folder, min_count = min(
        ((root, len(files)) for root, _,
            files in os.walk(directory) if len(files) > 0),
        key=lambda x: x[1],
        default=(None, 0)
    )
    print(f"Minor folder: {min_folder}")
    print(f"# Files: {min_count}")

    # Process subdirectories
    for root, _, files in os.walk(dest_dir):
        print("Processing:", root)
        for file in files:
            if is_jpg(file):
                file_path = os.path.join(root, file)
                aug(file_path, "dir")

        # Delete newest files if count exceeds n
        files_in_root = os.listdir(root)
        jpg_files = [os.path.join(root, f) for f in files_in_root if is_jpg(f)]
        jpg_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        max_allowed = min_count * 6

        if len(jpg_files) > max_allowed:
            excess_files = jpg_files[max_allowed:]
            print("Removing excess files...")
            for file in excess_files:
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"Erro ao remover {file}: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Augmentation.py [path_to_image] or "
              "[path_to_directory]")
        sys.exit(1)

    cli_arg = sys.argv[1]
    if os.path.isfile(cli_arg):
        aug(cli_arg, "file")
    elif os.path.isdir(cli_arg):
        process_directory(cli_arg)
    else:
        print(f"Invalid argument: {cli_arg}")
