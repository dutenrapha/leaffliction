import os
import random
import shutil
from pathlib import Path

def split_test_dataset(source_dir: str, test_dir: str, percentage: float):
    """
    Divide uma porcentagem de imagens de subdiretórios em um diretório de origem para um diretório de teste,
    mantendo a estrutura de pastas.

    :param source_dir: Caminho do diretório de origem contendo subdiretórios de imagens.
    :param test_dir: Caminho do diretório de destino para as imagens de teste.
    :param percentage: Porcentagem de imagens a serem movidas para o diretório de teste (0 a 1).
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

    print(f"Divisão concluída. Imagens de teste movidas para '{test_dir}'.")

split_test_dataset("images/grape", "test_dataset_grape", 0.2)
split_test_dataset("images/apple", "test_dataset_apple", 0.2)