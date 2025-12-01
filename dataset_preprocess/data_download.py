from pathlib import Path


def create_subset(data_dir: str = "../dataset", num_images: int = 800) -> None:
    data_dir = Path(data_dir)
    subset_dir = data_dir / "subset"
    subset_dir.mkdir(exist_ok=True)

    # Using only validation split (5k images)
    val_dir = data_dir / "val2017"

    print("=" * 60)
    print("Creating subset files...")
    print(f"data_dir   : {data_dir.resolve()}")
    print(f"num_images : {num_images}")
    print("=" * 60)

    if not val_dir.exists():
        raise FileNotFoundError(f"val2017 directory not found at: {val_dir}")

    all_val_images = sorted(val_dir.glob("*.jpg"))
    if not all_val_images:
        raise RuntimeError(f"No .jpg files found in {val_dir}")

    num_total = min(len(all_val_images), num_images)
    all_val_images = all_val_images[:num_total]

    split_idx = int(num_total * 0.8)
    train_images = all_val_images[:split_idx]
    val_images = all_val_images[split_idx:]

    # Write train subset
    train_file = subset_dir / "train_subset.txt"
    with train_file.open("w") as f:
        for img in train_images:
            f.write(f"{img.name}\n")
    print(f"Created {train_file} with {len(train_images)} images.")

    # Write val subset
    val_file = subset_dir / "val_subset.txt"
    with val_file.open("w") as f:
        for img in val_images:
            f.write(f"{img.name}\n")
    print(f"Created {val_file} with {len(val_images)} images.")

    print("\nSummary:")
    print(f"  Total images used: {num_total}")
    print(f"  Train: {len(train_images)}")
    print(f"  Val  : {len(val_images)}")


def verify_dataset(data_dir: str = "./dataset") -> bool:
    """
    Quick sanity check that required folders exist.
    """
    data_dir = Path(data_dir)
    val_dir = data_dir / "val2017"

    print("\nVerifying dataset folders...")
    ok = True

    if val_dir.exists():
        print(f"Found val2017 at {val_dir}")
    else:
        print(f"Missing val2017 at {val_dir}")
        ok = False

    if ok:
        print("All required folders are present.")
    else:
        print("Some required folders are missing.")

    return ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create COCO subset txt files from existing folders."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../dataset",
        help="Root directory containing val2017 folder.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4993,
        help="Number of images to use for subsets (default: 100).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify that required folders exist, then exit.",
    )

    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.data_dir)
    else:
        create_subset(
            data_dir=args.data_dir,
            num_images=args.num_images,
        )
