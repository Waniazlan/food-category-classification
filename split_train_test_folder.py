import argparse
import shutil
from pathlib import Path


DEFAULT_CLASSES = [
    "sushi",
    "cheesecake",
    "ice_cream",
    "fried_rice",
    "hamburger",
]


def read_split_file(path):
    class_to_images = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            class_name, image_id = line.strip().split("/")
            class_to_images.setdefault(class_name, []).append(f"{image_id}.jpg")
    return class_to_images


def copy_class_images(source_dir, destination_dir, class_name, filenames, limit=None):
    class_source = source_dir / class_name
    class_destination = destination_dir / class_name
    class_destination.mkdir(parents=True, exist_ok=True)

    selected_files = filenames[:limit] if limit else filenames
    copied = 0
    for filename in selected_files:
        source_file = class_source / filename
        if source_file.exists():
            shutil.copy2(source_file, class_destination / filename)
            copied += 1

    return copied


def prepare_subset(food101_dir, output_dir, classes, train_limit, test_limit):
    images_dir = food101_dir / "images"
    meta_dir = food101_dir / "meta"

    if not images_dir.is_dir() or not meta_dir.is_dir():
        raise FileNotFoundError(
            "Expected Food-101 dataset at food-101/images and food-101/meta. "
            "Download and extract food-101.tar.gz first."
        )

    train_map = read_split_file(meta_dir / "train.txt")
    test_map = read_split_file(meta_dir / "test.txt")

    selected_classes = set(classes)
    for split in ("train", "test"):
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir() and class_dir.name not in selected_classes:
                shutil.rmtree(class_dir)

    for class_name in classes:
        if class_name not in train_map or class_name not in test_map:
            raise ValueError(f"Class '{class_name}' is not present in Food-101 metadata.")

        train_count = copy_class_images(
            images_dir,
            output_dir / "train",
            class_name,
            train_map[class_name],
            train_limit,
        )
        test_count = copy_class_images(
            images_dir,
            output_dir / "test",
            class_name,
            test_map[class_name],
            test_limit,
        )
        print(f"{class_name}: {train_count} train, {test_count} test images")

    print(f"\nPrepared dataset at: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a smaller train/test Food-101 subset for CNN classification."
    )
    parser.add_argument("--source", default="food-101", help="Path to extracted Food-101 dataset.")
    parser.add_argument("--output", default="food-5", help="Output directory for the 5-class dataset.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Food-101 class names to include.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="Maximum training images per class. Use 0 for all available training images.",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=0,
        help="Maximum test images per class. Use 0 for all available test images.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_subset(
        food101_dir=Path(args.source),
        output_dir=Path(args.output),
        classes=args.classes,
        train_limit=args.train_limit or None,
        test_limit=args.test_limit or None,
    )
