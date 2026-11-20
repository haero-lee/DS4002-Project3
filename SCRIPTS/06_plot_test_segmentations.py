"""
===============================================================================
06_plot_test_segmentations.py
===============================================================================
PURPOSE
    Apply the trained 2D U-Net (best_model.keras) to every held-out TEST mouse
    and save one PNG per mouse comparing the model's LV segmentation with the
    expert segmentation. Each PNG has two rows - end-diastole (ED) and
    end-systole (ES), the two frames used for scoring - and four panels per row:

        1. Magnitude   - the MRI image (mid-ventricular slice)
        2. GT          - expert (ground-truth) segmentation
        3. Predicted   - the model's segmentation
        4. Overlap     - purple = GT and prediction agree (same structure)
                         red    = labeled in GT, missed or mislabeled by the model
                         blue   = predicted by the model, background in GT

    In panels 2 and 3, yellow = myocardium and green = blood pool.

    It also writes test_segmentation_summary.csv with each test mouse's
    whole-stack Dice at ED and ES (same calculation as the test evaluation in
    train_unet2d.py) and how many slices are labeled in each frame.

STANDALONE
    This script needs NO other project files: the few helper functions it uses
    from the original authors' utility.py (image normalisation, ED/ES detection,
    island removal) are copied in below unchanged in behaviour, and the 25 test
    mice are listed in the script. utility.py is (c) the mrphys authors,
    GPL-3.0: https://github.com/mrphys/Open-Source_Pre-Clinical_Segmentation

PROJECT CONTEXT
    DS 4002 Project 3: 2D U-Net vs. UNet 3+ for mouse LV segmentation.

BEFORE RUNNING
    1. Download the MRI data into DATA/WBH5Data (see DATA/DATA.txt).
    2. The CONFIGURATION section below already points to the reported run
       (OUTPUT/UNet2D_2026-11-16_17-16-24) and DATA/WBH5Data, relative to the
       repository root. Edit it only if your files are elsewhere:
        MODEL_PATH  path to best_model.keras
        DATA_DIR    folder containing Mouse1.h5 ... Mouse130.h5
        OUT_DIR     where to save the figures and CSV
        GPU         GPU index to use ("0", "1", ...) or None for the default

REQUIREMENTS
    pip install tensorflow h5py numpy matplotlib scikit-image

HOW TO RUN (from the repository root)
    python SCRIPTS/06_plot_test_segmentations.py

OUTPUTS  ->  OUT_DIR (OUTPUT/UNet2D_2026-11-16_17-16-24/test_segmentations/)
    <MouseID>_ED_ES.png              one figure per test mouse
    test_segmentation_summary.csv    whole-stack ED/ES Dice and labeled-slice counts

NOTES
    - Slice and frame numbers in figures and the CSV are 1-based.
    - The Dice in each Overlap title is for that single slice; the figure title
      gives the whole-stack Dice used in our reported results. These should
      match eval/test_dice_per_mouse.csv from training.
    - "partly_labeled_ES = True" means the ES frame has fewer labeled slices
      than ED; the model's (possibly correct) predictions on unlabeled slices
      then count as errors and lower the ES Dice.

AUTHOR / DATE
    Haero Lee (DS 4002, Project 3), November 2026
===============================================================================
"""

# ============================================================================
# CONFIGURATION - edit these before running
# ============================================================================
# Paths are relative to the repository root (run the script from there).
RUN_DIR = "OUTPUT/UNet2D_2026-11-16_17-16-24"               # reported training run
MODEL_PATH = f"{RUN_DIR}/best_model.keras"                  # trained model
DATA_DIR = "DATA/WBH5Data"                                  # folder with Mouse*.h5 files
OUT_DIR = f"{RUN_DIR}/test_segmentations"                   # output folder (created if needed)
GPU = None                                                  # GPU index ("0", "1", ...) or None
CROP = True                                     # zoom on the LV (False = full 256 x 256)
ISLAND_REMOVAL = True                           # same clean-up as the test evaluation

# Held-out test mice (90/15/25 split, seed 42; same as the UNet 3+ paper)
TEST_MICE = ["Mouse6", "Mouse10", "Mouse12", "Mouse15", "Mouse29", "Mouse36", "Mouse46",
             "Mouse49", "Mouse56", "Mouse60", "Mouse65", "Mouse71", "Mouse74", "Mouse75",
             "Mouse78", "Mouse79", "Mouse87", "Mouse89", "Mouse91", "Mouse99", "Mouse100",
             "Mouse111", "Mouse117", "Mouse118", "Mouse125"]

# ============================================================================
# Imports
# ============================================================================
import os   # file paths and environment variables

# Choose the GPU BEFORE TensorFlow is imported
if GPU is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide routine TensorFlow messages

import csv                                  # write the summary table
import matplotlib
matplotlib.use("Agg")                       # save figures to files; no window needed
import matplotlib.pyplot as plt
from matplotlib.patches import Patch        # colour boxes for the legend
import h5py                                 # read the .h5 data files
import numpy as np                          # array math
from skimage.measure import label, regionprops   # find and measure blobs in masks
import tensorflow as tf                     # load and run the trained model

# ============================================================================
# Colours (RGB, 0-1)
# ============================================================================
COL_MYO = (0.93, 0.63, 0.00)       # yellow = myocardium   (GT / Predicted panels)
COL_BLOOD = (0.11, 0.69, 0.48)     # green  = blood pool   (GT / Predicted panels)
COL_BOTH = (0.55, 0.31, 0.75)      # purple = agree        (Overlap panel)
COL_GT_ONLY = (0.88, 0.24, 0.19)   # red    = GT only      (Overlap panel)
COL_PRED_ONLY = (0.16, 0.47, 0.84) # blue   = predicted only (Overlap panel)
ALPHA = 0.6                        # transparency of coloured overlays
CROP_HALF = 56                     # crop size = 2 x 56 = 112 pixels around the LV


# ============================================================================
# Helper functions copied from the authors' utility.py (same behaviour)
# ============================================================================
def normalise_images(images):
    """Per-image min-max scaling to [0, 1] for a stack (N, H, W). Flat images -> 0."""
    arr = np.asarray(images, dtype=np.float32)
    mins = arr.min(axis=(-2, -1), keepdims=True)    # darkest pixel of each image
    maxs = arr.max(axis=(-2, -1), keepdims=True)    # brightest pixel of each image
    denom = maxs - mins
    out = np.zeros_like(arr, dtype=np.float32)
    np.divide(arr - mins, denom, out=out, where=denom > 0)
    return out


def bloodpool_volume_3d(mask_hws, min_component_area=50, require_overlap=None):
    """
    Blood pool size (pixels) across all slices of one frame, ignoring blobs
    smaller than min_component_area and, optionally, pixels outside the ED LV.
    Used to find end-systole (smallest blood pool).
    """
    vol = 0
    for s in range(mask_hws.shape[2]):
        sl = (mask_hws[..., s] == 2)                 # blood pool on this slice
        lab = label(sl)                              # separate blobs
        keep = np.zeros_like(sl, dtype=bool)
        for rp in regionprops(lab):
            if rp.area >= min_component_area:        # drop tiny blobs
                keep[lab == rp.label] = True
        sl = keep
        if require_overlap is not None:
            sl = sl & require_overlap[..., s]        # only inside the ED LV outline
        vol += int(sl.sum())
    return vol


def pick_ed_es(mask_hwsf, roi_frames):
    """
    ED = first labeled frame; ES = labeled frame (other than ED) with the
    smallest blood pool (ties -> earliest). Returns (ED, ES); ES = -1 if none.
    """
    roi_frames = sorted(int(f) for f in roi_frames)
    if not roi_frames:
        return -1, -1
    ed = roi_frames[0]
    ed_footprint = (mask_hwsf[..., ed] == 2) | (mask_hwsf[..., ed] == 1)
    vols = {f: bloodpool_volume_3d(mask_hwsf[..., f], 50, ed_footprint) for f in roi_frames}
    others = [f for f in roi_frames if f != ed]
    es = min(others, key=lambda f: (vols[f], f)) if others else -1
    return ed, es


def island_removal(seg, min_slices=2, min_area=10, distance_threshold=40):
    """
    For each class, keep only 3D blobs that span >= min_slices slices and lie
    within distance_threshold of the main blob (the one spanning the most
    slices); delete stray blobs. seg is (H, W, S).
    """
    cleaned = np.zeros_like(seg, dtype=np.uint8)
    for class_label in (1, 2):                       # myocardium, blood pool
        labels_3d = label((seg == class_label).astype(np.uint8), connectivity=1)
        regions = [r for r in regionprops(labels_3d) if r.area >= min_area]
        if not regions:
            continue
        dominant = max(regions, key=lambda r: len(set(c[2] for c in r.coords)))
        centre = np.array(dominant.centroid)
        for r in regions:
            n_slices = len(set(c[2] for c in r.coords))
            dist = np.linalg.norm(np.array(r.centroid) - centre)
            if n_slices >= min_slices and dist <= distance_threshold:
                cleaned[tuple(r.coords.T)] = class_label
    return cleaned


# ============================================================================
# Other helpers
# ============================================================================
def colour_layer(rgb_by_mask, shape):
    """Transparent RGBA image from a list of (boolean mask, colour)."""
    layer = np.zeros((*shape, 4))
    for mask, rgb in rgb_by_mask:
        layer[mask] = (*rgb, ALPHA)
    return layer


def dice(gt, pred):
    """Dice = 2|A∩B| / (|A|+|B|); NaN if both masks are empty."""
    total = gt.sum() + pred.sum()
    return np.nan if total == 0 else 2.0 * np.logical_and(gt, pred).sum() / total


def fmt(d):
    """Format a Dice value for titles ('n/a' when undefined)."""
    return "n/a" if np.isnan(d) else f"{d:.3f}"


def n_labeled_slices(msk, frame):
    """Number of slices in a frame that contain any expert label."""
    return int(sum(msk[:, :, s, frame].any() for s in range(msk.shape[2])))


def predict_frame(model, img, frame):
    """
    Segment every slice of one frame, exactly as in the test evaluation.
    Returns (normalised images (S, H, W), predicted labels (H, W, S)).
    """
    # (H, W, S) -> (S, H, W), then scale each slice to [0, 1]
    norm = normalise_images(np.moveaxis(img[:, :, :, frame], -1, 0).astype(np.float32))
    # All slices of the frame go through the model as one batch
    probs = model.predict(norm[..., None], verbose=0)
    if isinstance(probs, (list, tuple)):        # deep-supervision models return a list
        probs = probs[-1]
    # Most likely class per pixel, back to (H, W, S)
    pred = np.moveaxis(np.argmax(probs, axis=-1).astype(np.uint8), 0, -1)
    if ISLAND_REMOVAL:
        pred = island_removal(pred)
    return norm, pred


def draw_row(axes, image, gt, pr, row_info, dice_info):
    """Draw the four panels (magnitude, GT, predicted, overlap) on one row."""
    agree = (gt > 0) & (gt == pr)              # same structure in both -> purple
    gt_only = (gt > 0) & (gt != pr)            # in GT, missed or mislabeled -> red
    pred_only = (pr > 0) & (gt == 0)           # predicted where GT is background -> blue

    # Optional crop centred on the LV (GT and prediction combined)
    region = (gt > 0) | (pr > 0)
    H, W = gt.shape
    if CROP and region.any():
        ys, xs = np.nonzero(region)
        y0 = min(max(int(ys.mean()) - CROP_HALF, 0), H - 2 * CROP_HALF)
        x0 = min(max(int(xs.mean()) - CROP_HALF, 0), W - 2 * CROP_HALF)
        sl = (slice(y0, y0 + 2 * CROP_HALF), slice(x0, x0 + 2 * CROP_HALF))
    else:
        sl = (slice(0, H), slice(0, W))
    image, gt, pr = image[sl], gt[sl], pr[sl]
    agree, gt_only, pred_only = agree[sl], gt_only[sl], pred_only[sl]

    # Panel 1: magnitude
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(row_info, fontsize=10)
    # Panel 2: ground truth
    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(colour_layer([(gt == 1, COL_MYO), (gt == 2, COL_BLOOD)], gt.shape))
    axes[1].set_title("Ground truth (expert)\n", fontsize=10)
    # Panel 3: prediction
    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(colour_layer([(pr == 1, COL_MYO), (pr == 2, COL_BLOOD)], pr.shape))
    axes[2].set_title("Predicted (2D U-Net)\n", fontsize=10)
    # Panel 4: overlap
    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(colour_layer([(agree, COL_BOTH), (gt_only, COL_GT_ONLY),
                                 (pred_only, COL_PRED_ONLY)], agree.shape))
    axes[3].set_title(f"Overlap\n{dice_info}", fontsize=10)
    for ax in axes:
        ax.axis("off")


# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Load the model once; compile=False skips the training-only loss/metrics
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"Loaded model: {MODEL_PATH}")
    print(f"{len(TEST_MICE)} test mice\n")

    summary = []   # one row per mouse for the CSV
    for mouse_id in TEST_MICE:
        # ---- 1. Load this mouse ------------------------------------------
        with h5py.File(os.path.join(DATA_DIR, f"{mouse_id}.h5"), "r") as hf:
            img = hf["Images"][:]                   # (256, 256, slices, frames)
            msk = hf["Masks"][:]                    # same shape; 0/1/2 labels
            roi_frames = hf["Frames_with_ROI"][:]   # frames with expert labels
        S = img.shape[2]

        # ---- 2. ED and ES frames (same rule as the test evaluation) -------
        ed, es = pick_ed_es(msk, roi_frames)
        phases = [("ED", ed)] + ([("ES", es)] if es >= 0 else [])

        row = {"mouse": mouse_id}
        fig, axes = plt.subplots(len(phases), 4, figsize=(13, 3.6 * len(phases) + 0.8),
                                 dpi=150, squeeze=False)
        title_parts = []

        for r, (phase, f) in enumerate(phases):
            # ---- 3. Predict every slice of this frame --------------------
            norm, pred = predict_frame(model, img, f)
            gt_frame = msk[:, :, :, f]

            # ---- 4. Whole-stack Dice (same as the reported evaluation) ----
            stack_myo = dice(gt_frame == 1, pred == 1)
            stack_blood = dice(gt_frame == 2, pred == 2)
            row.update({f"{phase}_frame": f + 1,
                        f"{phase}_labeled_slices": f"{n_labeled_slices(msk, f)}/{S}",
                        f"{phase}_dice_myocardium": round(stack_myo, 4),
                        f"{phase}_dice_blood_pool": round(stack_blood, 4)})
            title_parts.append(f"{phase}: myo {fmt(stack_myo)}, blood {fmt(stack_blood)}")

            # ---- 5. Mid-ventricular slice = largest expert blood pool -----
            if gt_frame.any():
                s = int(np.argmax([(gt_frame[:, :, k] == 2).sum() for k in range(S)]))
            else:
                s = int(np.argmax([(pred[:, :, k] > 0).sum() for k in range(S)]))
            gt, pr = gt_frame[:, :, s], pred[:, :, s]

            # ---- 6. Draw this row ------------------------------------------
            row_info = f"{phase} magnitude\nframe {f + 1}, slice {s + 1} of {S}"
            dice_info = (f"slice Dice: myo {fmt(dice(gt == 1, pr == 1))}, "
                         f"blood {fmt(dice(gt == 2, pr == 2))}")
            draw_row(axes[r], norm[s], gt, pr, row_info, dice_info)

        # Flag a partly labeled ES frame (fewer labeled slices than ED)
        partial = es >= 0 and n_labeled_slices(msk, es) < n_labeled_slices(msk, ed)
        row["partly_labeled_ES"] = partial
        summary.append(row)

        # ---- 7. Figure title, legend, spacing, save ------------------------
        warn = "   [ES frame only partly labeled]" if partial else ""
        fig.suptitle(f"{mouse_id}  |  whole-stack Dice   " + "   ".join(title_parts) + warn,
                     fontsize=11, y=0.97)
        fig.legend(handles=[Patch(color=COL_MYO, label="Myocardium"),
                            Patch(color=COL_BLOOD, label="Blood pool"),
                            Patch(color=COL_BOTH, label="Overlap: both"),
                            Patch(color=COL_GT_ONLY, label="Overlap: GT only"),
                            Patch(color=COL_PRED_ONLY, label="Overlap: predicted only")],
                   loc="lower center", ncol=5, frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, 0.0))
        fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.07,
                            wspace=0.04, hspace=0.32)
        out = os.path.join(OUT_DIR, f"{mouse_id}_ED_ES.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"{mouse_id:<9} " + "   ".join(title_parts) +
              ("   (ES partly labeled)" if partial else "") + f"  -> {out}")

    # ---- 8. Summary CSV ----------------------------------------------------
    csv_path = os.path.join(OUT_DIR, "test_segmentation_summary.csv")
    fields = list(dict.fromkeys(k for r in summary for k in r))   # all columns, in order
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    print(f"\nSaved {len(summary)} figures and {csv_path}")


if __name__ == "__main__":
    main()
