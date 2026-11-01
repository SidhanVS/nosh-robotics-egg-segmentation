from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor

PROJECT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "sam_vit_b_01ec64.pth"

IMAGE_NAME = (
    "not_damaged_10_jpg.rf."
    "2347ffa54eea250ab240def10a3eb78f.jpg"
)
IMAGE_PATH = (
    PROJECT_DIR / "dataset" / "egg.v1i.folder" /
    "train" / "egg" / IMAGE_NAME
)
OUTPUT_DIR = PROJECT_DIR / "results" / "point_prompt_review_10_positive_negative"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    image_bgr = cv2.imread(str(IMAGE_PATH))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Selected device:", device)

    sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT_PATH))
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)
    print("SAM loaded and image prepared.")

    print("\nSelect exactly THREE points in this order:")
    print("  CLICK 1 — POSITIVE: inside the visible part of the egg.")
    print("  CLICK 2 — NEGATIVE: on a nearby leaf or stem, outside the egg.")
    print("  CLICK 3 — NEGATIVE: on another nearby leaf/stem, outside the egg.")
    print("Avoid the egg boundary. Press ENTER after three clicks.")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(image_rgb)
    ax.set_title(
        "1: Egg (positive) | 2: Leaf/stem (negative) | "
        "3: Another leaf/stem (negative)"
    )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("on")
    fig.tight_layout()

    points = plt.ginput(n=3, timeout=0, show_clicks=True)
    plt.close(fig)
    if len(points) != 3:
        raise RuntimeError(
            f"Expected 3 clicks but received {len(points)}. "
            "Run again and select all three points."
        )

    point_coords = np.asarray(points, dtype=np.float32)
    point_labels = np.asarray([1, 0, 0], dtype=np.int32)

    print("\nSelected points (x, y):")
    print("  Positive egg point:", point_coords[0].tolist())
    print("  Negative foliage point 1:", point_coords[1].tolist())
    print("  Negative foliage point 2:", point_coords[2].tolist())

    pd.DataFrame({
        "point_type": [
            "positive_egg", "negative_foliage_1", "negative_foliage_2"
        ],
        "x": point_coords[:, 0],
        "y": point_coords[:, 1],
        "label": point_labels
    }).to_csv(OUTPUT_DIR / "prompt_points.csv", index=False)

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    if masks is None or len(masks) == 0:
        raise RuntimeError("SAM returned no candidate masks.")

    fig, axes = plt.subplots(
        1, len(masks) + 1,
        figsize=(5 * (len(masks) + 1), 5)
    )
    axes = np.asarray(axes).reshape(-1)

    axes[0].imshow(image_rgb)
    axes[0].scatter(
        point_coords[0, 0], point_coords[0, 1],
        color="red", marker="+", s=150, linewidths=2,
        label="Positive: egg"
    )
    axes[0].scatter(
        point_coords[1:, 0], point_coords[1:, 1],
        color="yellow", marker="x", s=110, linewidths=2,
        label="Negative: foliage"
    )
    axes[0].set_title("Original + prompt points")
    axes[0].legend(loc="lower left", fontsize=7)
    axes[0].axis("off")

    for index, (mask, score) in enumerate(zip(masks, scores), start=1):
        binary_mask = mask.astype(bool)
        mask_u8 = binary_mask.astype(np.uint8) * 255
        mask_path = OUTPUT_DIR / f"candidate_{index}_mask.png"
        if not cv2.imwrite(str(mask_path), mask_u8):
            raise OSError(f"Could not save mask: {mask_path}")

        overlay = image_rgb.copy()
        pixels = binary_mask
        if np.any(pixels):
            green = np.zeros_like(overlay[pixels])
            green[:, 1] = 255
            overlay[pixels] = (
                0.4 * overlay[pixels] + 0.6 * green
            ).astype(np.uint8)

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        overlay_path = OUTPUT_DIR / f"candidate_{index}_overlay.png"
        if not cv2.imwrite(str(overlay_path), overlay_bgr):
            raise OSError(f"Could not save overlay: {overlay_path}")

        axes[index].imshow(overlay)
        axes[index].scatter(
            point_coords[0, 0], point_coords[0, 1],
            color="red", marker="+", s=150, linewidths=2
        )
        axes[index].scatter(
            point_coords[1:, 0], point_coords[1:, 1],
            color="yellow", marker="x", s=110, linewidths=2
        )
        axes[index].set_title(
            f"Candidate {index}\n"
            f"SAM score: {float(score):.3f} | "
            f"Area: {int(binary_mask.sum())} px"
        )
        axes[index].axis("off")

        print(
            f"Candidate {index}: SAM score={float(score):.4f}, "
            f"area={int(binary_mask.sum())} pixels"
        )
        print("  Mask:", mask_path)
        print("  Overlay:", overlay_path)

    fig.suptitle(
        "SAM Point-Prompt Candidates — Image 10 "
        "(1 positive, 2 negative foliage points)",
        fontsize=13
    )
    fig.tight_layout()
    comparison_path = OUTPUT_DIR / "point_prompt_negative_comparison.png"
    fig.savefig(comparison_path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("\nComparison saved:", comparison_path)
    print("Prompt coordinates saved:", OUTPUT_DIR / "prompt_points.csv")
    print("Output folder:", OUTPUT_DIR)
    print(
        "\nReview all three masks visually. SAM score alone does not "
        "establish that a mask is a correct egg segmentation."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
