
from pathlib import Path
import pandas as pd

# 1. Define paths

project_dir = Path(__file__).resolve().parents[1]

csv_path = (
    project_dir / "results" /
    "automatic_sam_candidates" /
    "candidate_metadata.csv"
)

# 2. Read candidate metadata

df = pd.read_csv(csv_path)

# 3. Calculate geometric properties

df["aspect_ratio"] = (
    df["bbox_width"] / df["bbox_height"].clip(lower=1)
)

df["image_area"] = 640 * 640

df["area_ratio"] = (
    df["area"] / df["image_area"]
)

# 4. Display overall statistics

print("Total candidate masks:", len(df))

print("\nAREA RATIO STATISTICS:")
print(df["area_ratio"].describe())

print("\nASPECT RATIO STATISTICS:")
print(df["aspect_ratio"].describe())

# 5. Inspect candidates from one image

selected_image = (
    "not_damaged_103_jpg.rf."
    "c0bd961caec5acf3a6582c9cb2551634.jpg"
)

image_candidates = df[
    df["image_name"] == selected_image
].copy()

image_candidates = image_candidates.sort_values(
    "area",
    ascending=False
)

print("\nCANDIDATES FOR SELECTED IMAGE:")

print(
    image_candidates[
        [
            "mask_id",
            "area",
            "bbox_width",
            "bbox_height",
            "aspect_ratio",
            "area_ratio",
            "predicted_iou",
            "stability_score"
        ]
    ].to_string(index=False)
)

# 6. Save the analysis

output_path = (
    project_dir / "results" /
    "candidate_geometry_analysis.csv"
)

df.to_csv(output_path, index=False)

print("\nAnalysis saved to:", output_path)
