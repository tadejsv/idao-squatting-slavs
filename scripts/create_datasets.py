import random
import shutil
from pathlib import Path

from tqdm.auto import tqdm

from src.data import IDAOData


def create_datasets():
    """ Create train, val, test and test_holdout from original idao_dataset/train."""
    full_dataset = IDAOData("idao_dataset/train")

    data_root = Path("data")
    data_root.mkdir(exist_ok=True)

    # Prepare new directories
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"
    test_holdout_dir = data_root / "test_holdout"

    data_exists = False
    try:
        train_dir.mkdir()
        val_dir.mkdir()
        test_dir.mkdir()
        test_holdout_dir.mkdir()
    except FileExistsError:
        data_exists = True

    random.seed(42)

    # Write files to new location
    if not data_exists:
        for file, (r_type, energy) in tqdm(
            zip(full_dataset.image_files, full_dataset.classes), desc="Copying files."
        ):
            if r_type == "ER" and energy in (1, 6, 20):
                shutil.copy(file, test_holdout_dir)
            elif r_type == "NR" and energy in (3, 10, 30):
                shutil.copy(file, test_holdout_dir)
            else:
                out_dir = random.choices(
                    [train_dir, val_dir, test_dir], [0.9, 0.05, 0.05]
                )
                shutil.copy(file, out_dir[0])
    else:
        print("Data already exists.")


if __name__ == "__main__":
    create_datasets()
