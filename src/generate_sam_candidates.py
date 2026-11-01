
from pathlib import Path
import csv

import cv2
import torch

from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator
)


# 1. Define paths

project_dir = Path(__file__).resolve().parents[1]

checkpoint_path = (
    project_dir / "checkpoints" / "sam_vit_b_01ec64.pth"
)

dataset_dir = (
    project_dir / "dataset" / "egg.v1i.folder"
)

output_dir = (
    project_dir / "results" / "automatic_sam_candidates"
)

output_dir.mkdir(parents=True, exist_ok=True)

# 2. Collect images from all dataset splits

dataset_splits = ["train", "valid", "test"]

image_extensions = {
    ".jpg", ".jpeg", ".png"
}

image_paths = []

for split in dataset_splits:

    split_dir = dataset_dir / split

    split_images = sorted(
        path for path in split_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in image_extensions
    )

    image_paths.extend(split_images)

    print(f"{split.capitalize()} images:", len(split_images))

print("Total images selected:", len(image_paths))

# 3. Load the pretrained SAM model

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Selected device:", device)

sam = sam_model_registry["vit_b"](
    checkpoint=str(checkpoint_path)
)

sam.to(device=device)

print("SAM model loaded successfully!")


# 4. Configure automatic mask generation

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    points_per_batch=4,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=0,
    min_mask_region_area=100
)

print("Automatic mask generator ready!")


# 5. Prepare a CSV file for mask information

csv_path = output_dir / "candidate_metadata.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "image_name",
        "mask_id",
        "area",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "predicted_iou",
        "stability_score"
    ])


    # 6. Process each image

    for image_index, image_path in enumerate(
        image_paths, start=1
    ):

        print("\nProcessing:", image_path.name)

        image = cv2.imread(str(image_path))

        if image is None:
            print("Could not read image. Skipping.")
            continue

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        masks = mask_generator.generate(image_rgb)

        print("Candidate masks generated:", len(masks))

        # Create a separate folder for this image
        image_output_dir = (
            output_dir / image_path.stem
        )

        image_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Sort candidates by area
        masks = sorted(
            masks,
            key=lambda item: item["area"],
            reverse=True
        )

        # Save every candidate mask
        for mask_index, mask_data in enumerate(
            masks, start=1
        ):

            segmentation = mask_data["segmentation"]

            mask_image = (
                segmentation.astype("uint8") * 255
            )

            mask_filename = (
                f"mask_{mask_index:03d}.png"
            )

            mask_path = (
                image_output_dir / mask_filename
            )

            cv2.imwrite(
                str(mask_path),
                mask_image
            )

            x, y, width, height = mask_data["bbox"]

            writer.writerow([
                image_path.name,
                mask_index,
                mask_data["area"],
                x,
                y,
                width,
                height,
                mask_data["predicted_iou"],
                mask_data["stability_score"]
            ])

        print(
            "Saved", len(masks),
            "candidate masks for", image_path.name
        )

print("\nFull dataset processing completed!")
print("Candidate masks saved in:", output_dir)
print("Metadata saved in:", csv_path)
