from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd

# Final full-dataset run using the Version 3 experimental thresholds.
# This script does not modify candidate masks, metadata, or earlier results.

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_DIR / "dataset" / "egg.v1i.folder"
CANDIDATE_ROOT = PROJECT_DIR / "results" / "automatic_sam_candidates"
METADATA_PATH = CANDIDATE_ROOT / "candidate_metadata.csv"
OUTPUT_ROOT = PROJECT_DIR / "results" / "final_sam_selection_v3_all_110"

MIN_AREA_RATIO = 0.025
MAX_AREA_RATIO = 0.40
MIN_ASPECT_RATIO = 0.60
MAX_ASPECT_RATIO = 1.90
MIN_STABILITY = 0.90
MIN_BBOX_FILL_RATIO = 0.68
OVERLAP_THRESHOLD = 0.70

# Process the three labelled subsets and the two Unlabeled images separately.
SOURCE_FOLDERS = [
    ("train", "egg", "train"),
    ("train", "Unlabeled", "train_unlabeled"),
    ("valid", "egg", "valid"),
    ("test", "egg", "test"),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
REQUIRED_COLUMNS = {
    "image_name", "mask_id", "area", "bbox_width",
    "bbox_height", "stability_score"
}

def read_metadata():
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Candidate metadata not found: {METADATA_PATH}")
    df = pd.read_csv(METADATA_PATH)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    return df

def create_overlay(image, combined_mask):
    overlay = image.copy()
    pixels = combined_mask > 0
    if np.any(pixels):
        green = np.zeros_like(image[pixels])
        green[:, 1] = 255
        overlay[pixels] = cv2.addWeighted(
            image[pixels], 0.4, green, 0.6, 0
        )
    return overlay

def process_image(image_path, split, source_folder, group_name, metadata):
    image_name = image_path.name
    image = cv2.imread(str(image_path))
    if image is None:
        return {
            "split": split, "source_folder": source_folder,
            "group": group_name, "image_name": image_name,
            "total_candidates": 0, "filtered_candidates": 0,
            "selected_masks": 0, "selected_candidate_ids": "[]",
            "invalid_candidate_masks": 0, "status": "Original image unreadable"
        }

    height, width = image.shape[:2]
    image_area = height * width
    image_candidates = metadata[
        metadata["image_name"] == image_name
    ].copy()
    total_candidates = len(image_candidates)

    if total_candidates:
        image_candidates["area_ratio"] = image_candidates["area"] / image_area
        image_candidates["aspect_ratio"] = (
            image_candidates["bbox_width"] /
            image_candidates["bbox_height"].clip(lower=1)
        )
        image_candidates["bbox_fill_ratio"] = (
            image_candidates["area"] /
            (
                image_candidates["bbox_width"].clip(lower=1) *
                image_candidates["bbox_height"].clip(lower=1)
            )
        )

        candidates = image_candidates[
            (image_candidates["area_ratio"] >= MIN_AREA_RATIO) &
            (image_candidates["area_ratio"] <= MAX_AREA_RATIO) &
            (image_candidates["aspect_ratio"] >= MIN_ASPECT_RATIO) &
            (image_candidates["aspect_ratio"] <= MAX_ASPECT_RATIO) &
            (image_candidates["stability_score"] >= MIN_STABILITY) &
            (image_candidates["bbox_fill_ratio"] >= MIN_BBOX_FILL_RATIO)
        ].copy()
    else:
        candidates = image_candidates

    mask_dir = CANDIDATE_ROOT / image_path.stem
    selected_ids = []
    selected_masks = []
    invalid_masks = 0

    for _, row in candidates.sort_values(
        "stability_score", ascending=False
    ).iterrows():
        mask_id = int(row["mask_id"])
        mask_path = mask_dir / f"mask_{mask_id:03d}.png"
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask_gray is None or mask_gray.shape[:2] != (height, width):
            invalid_masks += 1
            print(f"WARNING: Missing/invalid candidate mask: {mask_path}")
            continue

        binary_mask = mask_gray > 0
        if not np.any(binary_mask):
            invalid_masks += 1
            print(f"WARNING: Empty candidate mask: {mask_path}")
            continue

        duplicate = False
        if selected_masks:
            existing_union = np.zeros((height, width), dtype=bool)
            for previous_mask in selected_masks:
                existing_union |= previous_mask

            intersection = np.logical_and(
                binary_mask, existing_union
            ).sum()
            smaller_area = min(binary_mask.sum(), existing_union.sum())
            overlap = (
                intersection / smaller_area
                if smaller_area > 0 else 0.0
            )
            if overlap > OVERLAP_THRESHOLD:
                duplicate = True

        if not duplicate:
            selected_ids.append(mask_id)
            selected_masks.append(binary_mask)

    image_output_dir = (
        OUTPUT_ROOT / group_name / image_path.stem
    )
    image_output_dir.mkdir(parents=True, exist_ok=True)
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    for index, binary_mask in enumerate(selected_masks, start=1):
        individual_mask = binary_mask.astype(np.uint8) * 255
        mask_output = image_output_dir / f"egg_{index}_mask.png"
        if not cv2.imwrite(str(mask_output), individual_mask):
            raise OSError(f"Could not save individual mask: {mask_output}")
        combined_mask = cv2.bitwise_or(combined_mask, individual_mask)

    combined_path = image_output_dir / "combined_egg_mask.png"
    overlay_path = image_output_dir / "selection_overlay.png"

    if not cv2.imwrite(str(combined_path), combined_mask):
        raise OSError(f"Could not save combined mask: {combined_path}")
    if not cv2.imwrite(str(overlay_path), create_overlay(image, combined_mask)):
        raise OSError(f"Could not save overlay: {overlay_path}")

    status = "Processed" if selected_ids else "No masks selected"
    if total_candidates == 0:
        status = "No candidate metadata"
    if invalid_masks:
        status += f"; {invalid_masks} invalid candidate masks"

    return {
        "split": split,
        "source_folder": source_folder,
        "group": group_name,
        "image_name": image_name,
        "total_candidates": total_candidates,
        "filtered_candidates": len(candidates),
        "selected_masks": len(selected_ids),
        "selected_candidate_ids": str(selected_ids),
        "invalid_candidate_masks": invalid_masks,
        "status": status
    }

def main():
    print("=" * 72)
    print("SAM AUTOMATIC SELECTION — VERSION 3 FULL DATASET")
    print("=" * 72)
    print("Parameters:")
    print(
        f"  area_ratio={MIN_AREA_RATIO:.3f}–{MAX_AREA_RATIO:.2f}; "
        f"aspect_ratio={MIN_ASPECT_RATIO:.2f}–{MAX_ASPECT_RATIO:.2f}"
    )
    print(
        f"  stability>={MIN_STABILITY:.2f}; "
        f"bbox_fill_ratio>={MIN_BBOX_FILL_RATIO:.2f}; "
        f"overlap_threshold={OVERLAP_THRESHOLD:.2f}"
    )
    print("Output:", OUTPUT_ROOT)
    print("Earlier result folders will not be modified.")

    metadata = read_metadata()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    expected_source_folders = 0

    for split, source_folder, group_name in SOURCE_FOLDERS:
        source_dir = DATASET_ROOT / split / source_folder
        if not source_dir.is_dir():
            print(f"WARNING: Source folder not found: {source_dir}")
            continue

        images = sorted(
            path for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        expected_source_folders += len(images)
        print(f"\n--- {group_name.upper()}: {len(images)} images ---")

        for index, image_path in enumerate(images, start=1):
            result = process_image(
                image_path, split, source_folder, group_name, metadata
            )
            rows.append(result)
            print(
                f"[{group_name} {index}/{len(images)}] {image_path.name}: "
                f"candidates={result['total_candidates']}, "
                f"filtered={result['filtered_candidates']}, "
                f"selected={result['selected_masks']}, "
                f"IDs={result['selected_candidate_ids']}, "
                f"status={result['status']}"
            )

    summary = pd.DataFrame(rows)
    summary_path = OUTPUT_ROOT / "full_dataset_selection_summary_v3.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 72)
    print("VERSION 3 FULL-DATASET SELECTION COMPLETED")
    print("=" * 72)
    if not summary.empty:
        print(summary.groupby("group").agg(
            images_processed=("image_name", "count"),
            selected_masks=("selected_masks", "sum"),
            images_without_masks=(
                "selected_masks", lambda values: int((values == 0).sum())
            )
        ).to_string())
        print("\nTotal images discovered:", expected_source_folders)
        print("Total summary rows:", len(summary))
        print("Total selected masks:", int(summary["selected_masks"].sum()))
        print(
            "Images without selected masks:",
            int((summary["selected_masks"] == 0).sum())
        )
        print(
            "Images with invalid candidate masks:",
            int((summary["invalid_candidate_masks"] > 0).sum())
        )

    print("\nSummary CSV:", summary_path)
    print("Output folder:", OUTPUT_ROOT)
    if expected_source_folders != 110 or len(summary) != 110:
        print(
            "\nWARNING: Expected 110 images (including Unlabeled); "
            "check the split folders and image extensions."
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
