"""
===============================================================================
04_unet2d_utils.py
===============================================================================
PURPOSE
    Helper functions for training a plain 2D U-Net to segment the left
    ventricle (LV) in mouse cine short-axis (SAX) cardiac MRI. It segments two
    structures:
        - LV myocardium  (label 1)
        - LV blood pool  (label 2)
    Everything else is background (label 0).

    This file is imported by 03_train_unet2d.py. It is NOT meant to be run on its
    own (running it directly does nothing).

PROJECT CONTEXT
    DS 4002 Project 3: comparing a plain 2D U-Net against the published
    UNet 3+ from the mrphys "Open-Source Pre-Clinical Segmentation" repository
    (https://github.com/mrphys/Open-Source_Pre-Clinical_Segmentation).
    This file reproduces the released notebook's mouse assignments and is
    used together with 02_utility.py (derived from the authors' utility.py) for loading data,
    augmentation, and Dice evaluation.

WHAT IS IN THIS FILE
    1. set_seeds, setup_gpu        -> reproducibility and GPU configuration
    2. build_unet2d                -> the 2D U-Net model
    3. dice_myo, dice_blood,
       dice_fg, dice_ce_loss       -> training metrics and loss function
    4. get_augmentation            -> random image transforms for training
    5. split_by_mouse, PAPER_SPLIT,
       check_matches_paper,
       save_split, load_split,
       select_by_mice              -> mouse-level train/val/test split
    6. summarize_scores            -> mean ± SD of test Dice across mice

REQUIREMENTS
    Python 3.9+ (3.11 recommended), tensorflow (2.x), numpy, albumentations,
    opencv-python (imported as cv2).
    Install:  pip install "tensorflow[and-cuda]" numpy albumentations opencv-python

IMPORTANT NOTES FOR ANYONE USING THIS FILE
    - Images must be 256 x 256 with one channel, scaled to [0, 1].
    - Masks must contain only the integers 0, 1, 2.
    - The split is done by MOUSE, not by image, so slices from one heart never
      appear in two sets (this prevents data leakage).
    - With seed 42 and a 90/15/25 split, split_by_mouse() gives exactly the
      same mice as the released UNet 3+ notebook lists. This does not verify
      the final paper partition. Changing seed or split sizes changes the split.

AUTHOR / DATE
    Haero Lee (DS 4002, Project 3), November 2026
===============================================================================
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import os        # environment variables (used to fix Python's hash seed)
import json      # read/write the train/val/test split as a .json file
import random    # Python's built-in random number generator (seeded below)
from typing import Dict, List, Sequence, Tuple  # type hints for readability only

import numpy as np          # array math; also used to shuffle mice for the split
import tensorflow as tf     # deep learning framework used to build/train the model
import albumentations as A  # image augmentation library (flips, rotations, etc.)
import cv2                  # OpenCV; only used here for a rotation border setting

# ----------------------------------------------------------------------------
# Constants describing the segmentation labels
# ----------------------------------------------------------------------------
NUM_CLASSES = 3          # 0 = background, 1 = myocardium, 2 = blood pool
MYO, BLOOD = 1, 2        # channel/label index for each foreground structure


# ============================================================================
# 1. Reproducibility and GPU
# ============================================================================
def set_seeds(seed: int = 42) -> None:
    """
    Seed every random number generator we use so that a run can be repeated
    and give (as close as possible) the same result.
    """
    # Fix Python's hash seed (affects ordering of some built-in operations)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Seed Python's built-in random module
    random.seed(seed)
    # Seed NumPy (used for shuffling and by the augmentation generator)
    np.random.seed(seed)
    # Seed TensorFlow/Keras (weight initialisation, dropout, etc.).
    # This one call also re-seeds Python and NumPy internally.
    tf.keras.utils.set_random_seed(seed)


def setup_gpu(mixed_precision: bool = False) -> None:
    """
    Prepare the GPU(s) for training.

    - Turns on "memory growth" so TensorFlow only takes the GPU memory it
      actually needs, instead of grabbing the whole card. This matters on a
      shared lab machine where other people may be using the same GPU.
    - Optionally turns on mixed precision (16-bit math), which can speed up
      training on newer GPUs and lower memory use.

    Which GPU is used is chosen in 03_train_unet2d.py with --gpu, which sets
    CUDA_VISIBLE_DEVICES before TensorFlow is imported.
    """
    # Ask TensorFlow which GPUs it can see
    gpus = tf.config.list_physical_devices("GPU")
    # If none are found, warn the user: training will still run, but slowly on CPU
    if not gpus:
        print("WARNING: No GPU found - training will run on CPU.")
    # Enable memory growth on every visible GPU
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError as e:
            # This error happens if the GPU was already initialised; just report it
            print(e)
    # Print the GPUs being used so it shows up in the training log
    print(f"Visible GPUs: {[g.name for g in gpus]}")

    # Optional: use 16-bit floats for most computations
    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: ON (mixed_float16)")


# ============================================================================
# 2. Model
# ============================================================================
def _conv_block(x, filters: int, name: str, batch_norm: bool = True,
                depth: int = 2):
    """
    The basic building block of the U-Net:
        (3x3 convolution -> batch normalisation -> ReLU)  repeated `depth` times.

    x          : input tensor
    filters    : number of feature maps (channels) each convolution produces
    name       : prefix for layer names (makes model.summary() readable)
    batch_norm : whether to normalise activations after each convolution
    depth      : how many conv layers to stack (2, as in the original U-Net)
    """
    for i in range(depth):
        # 3x3 convolution. padding="same" keeps the image size unchanged.
        # HeNormal initialisation works well with ReLU; seed=0 makes it repeatable.
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same",
            kernel_initializer=tf.keras.initializers.HeNormal(seed=0),
            name=f"{name}_conv{i+1}")(x)
        # Batch normalisation keeps activations in a stable range -> faster,
        # more stable training
        if batch_norm:
            x = tf.keras.layers.BatchNormalization(name=f"{name}_bn{i+1}")(x)
        # ReLU non-linearity (sets negative values to 0)
        x = tf.keras.layers.Activation("relu", name=f"{name}_relu{i+1}")(x)
    return x


def build_unet2d(input_shape: Tuple[int, int, int] = (256, 256, 1),
                 filters: Sequence[int] = (32, 64, 128, 256, 512),
                 num_classes: int = NUM_CLASSES,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 name: str = "UNet2D") -> tf.keras.Model:
    """
    Build a plain 2D U-Net (Ronneberger et al., 2015).

    How it works:
      - Encoder (left side of the "U"): each level applies a conv block and
        then halves the image size with max pooling, so the network sees
        larger areas and learns WHAT is in the image.
      - Bottleneck: the smallest, deepest level (16 x 16 for a 256 x 256 input).
      - Decoder (right side): each level doubles the image size again and
        joins ("concatenates") the matching encoder output through a skip
        connection, which brings back fine detail about WHERE edges are.
      - Output: a 1x1 convolution + softmax gives, for every pixel, the
        probability of background / myocardium / blood pool.

    The filter widths are (32, 64, 128, 256, 512). Matching widths alone
    does not control model size or training conditions; the published
    benchmark comparison does not isolate an architectural effect.

    Returns a Keras Model: input (batch, 256, 256, 1) -> output (batch, 256, 256, 3)
    """
    # Input layer: one grayscale 256 x 256 image per sample
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = inputs
    # List that stores each encoder level's output for the skip connections
    skips = []

    # ---- Encoder: 4 levels (32, 64, 128, 256 filters) ----
    for lvl, f in enumerate(filters[:-1], start=1):
        # Learn features at this resolution
        x = _conv_block(x, f, name=f"enc{lvl}", batch_norm=batch_norm)
        # Save the output so the decoder can reuse it later
        skips.append(x)
        # Halve the height and width (256 -> 128 -> 64 -> 32 -> 16)
        x = tf.keras.layers.MaxPooling2D(2, name=f"enc{lvl}_pool")(x)

    # ---- Bottleneck: deepest level (512 filters, 16 x 16) ----
    x = _conv_block(x, filters[-1], name="bottleneck", batch_norm=batch_norm)
    # Optional dropout to reduce overfitting (off by default: dropout = 0.0)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout, name="bottleneck_dropout")(x)

    # ---- Decoder: go back up through the 4 levels in reverse order ----
    for lvl, (f, skip) in reversed(list(enumerate(zip(filters[:-1], skips), start=1))):
        # Transposed convolution doubles the height and width (learned upsampling)
        x = tf.keras.layers.Conv2DTranspose(f, 2, strides=2, padding="same",
                                            name=f"dec{lvl}_up")(x)
        # Skip connection: stack the upsampled features with the encoder
        # features from the SAME level (this is what makes it a U-Net)
        x = tf.keras.layers.Concatenate(name=f"dec{lvl}_concat")([x, skip])
        # Refine the combined features
        x = _conv_block(x, f, name=f"dec{lvl}", batch_norm=batch_norm)

    # ---- Output layer ----
    # 1x1 convolution maps the features to 3 scores per pixel (one per class)
    logits = tf.keras.layers.Conv2D(num_classes, 1, name="logits")(x)
    # Softmax turns the 3 scores into probabilities that sum to 1.
    # dtype="float32" keeps the output in full precision even if mixed
    # precision is turned on (needed for a numerically stable loss).
    outputs = tf.keras.layers.Activation("softmax", dtype="float32",
                                         name="softmax")(logits)
    # Wrap everything into a Keras Model object
    return tf.keras.Model(inputs, outputs, name=name)


# ============================================================================
# 3. Metrics and loss
# ============================================================================
# Dice coefficient = 2 * |overlap| / (|prediction| + |ground truth|).
# 1.0 = perfect overlap, 0.0 = no overlap. "Soft" Dice uses the predicted
# probabilities directly instead of hard 0/1 labels, so it can be used during
# training.

def dice_for_class(class_idx: int, name: str, smooth: float = 1e-6):
    """
    Create a soft Dice metric for ONE class (e.g. myocardium only).
    Computed over the whole batch during training/validation.

    The returned metric is named 'dice_myo' / 'dice_blood' on purpose:
    utility.save_loss_and_dice_plots() looks for those names when drawing the
    training curves.
    """
    def metric(y_true, y_pred):
        # Pull out the channel for this class (y_true is one-hot encoded)
        yt = tf.cast(y_true[..., class_idx], tf.float32)
        yp = tf.cast(y_pred[..., class_idx], tf.float32)
        # Overlap between ground truth and predicted probabilities
        inter = tf.reduce_sum(yt * yp)
        # Dice formula; `smooth` avoids dividing by zero on empty images
        return (2.0 * inter + smooth) / (tf.reduce_sum(yt) + tf.reduce_sum(yp) + smooth)
    # Give the function a name so Keras logs it as e.g. "dice_myo"/"val_dice_myo"
    metric.__name__ = name
    return metric


# Ready-made metrics for the two structures we care about
dice_myo = dice_for_class(MYO, "dice_myo")        # myocardium Dice
dice_blood = dice_for_class(BLOOD, "dice_blood")  # blood pool Dice


def dice_fg(y_true, y_pred):
    """
    Foreground Dice = average of myocardium and blood pool Dice.
    Its validation version (val_dice_fg) decides which epoch's model is saved
    as "best" and when early stopping kicks in.
    """
    return 0.5 * (dice_myo(y_true, y_pred) + dice_blood(y_true, y_pred))


def dice_ce_loss(y_true, y_pred, smooth: float = 1e-6, dice_weight: float = 1.0,
                 ce_weight: float = 1.0):
    """
    Loss function the model minimises during training:
        loss = (1 - mean foreground Dice) + categorical cross-entropy

    Why two terms?
      - Dice loss focuses on overlap of the small structures, so the model is
        not dominated by background (which is most of each 256 x 256 image).
      - Cross-entropy gives smooth, per-pixel feedback, which helps training
        get started and stay stable.
    """
    # Make sure both tensors are 32-bit floats
    y_true = tf.cast(y_true, tf.float32)
    # Keep probabilities slightly away from 0 and 1 so log() in cross-entropy
    # never sees exactly 0
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)

    # ---- Dice part (foreground classes only: channels 1 and 2) ----
    # Flatten every pixel of every image into rows; keep the 2 class columns
    yt = tf.reshape(y_true[..., 1:], [-1, NUM_CLASSES - 1])
    yp = tf.reshape(y_pred[..., 1:], [-1, NUM_CLASSES - 1])
    # Overlap and total size for each class
    inter = tf.reduce_sum(yt * yp, axis=0)
    denom = tf.reduce_sum(yt + yp, axis=0)
    # Average Dice over the two classes, turned into a loss (lower = better)
    dice_loss = 1.0 - tf.reduce_mean((2.0 * inter + smooth) / (denom + smooth))

    # ---- Cross-entropy part (all 3 classes, averaged over all pixels) ----
    ce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))

    # Combine the two (equal weights by default)
    return dice_weight * dice_loss + ce_weight * ce


# ============================================================================
# 4. Augmentation (training data only)
# ============================================================================
def get_augmentation() -> A.Compose:
    """
    Random changes applied to each training image on the fly, so the model
    sees slightly different versions every epoch and generalises better to
    new mice. Albumentations applies the SAME geometric change to the image
    and its mask, so the labels stay aligned with the heart.

    Used by utility.data_generator_with_augmentation(). Validation and test
    images are never augmented.
    """
    return A.Compose([
        # Mirror the image left-right half of the time
        A.HorizontalFlip(p=0.5),
        # Mirror the image top-bottom half of the time
        A.VerticalFlip(p=0.5),
        # Rotate by a random angle between -15 and +15 degrees half of the time;
        # areas rotated in from outside the image are filled with 0 (black)
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        # Slightly brighten/darken and change contrast (up to ±10%) half of the time,
        # imitating scan-to-scan intensity differences
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    ])


# ============================================================================
# 5. Mouse-level train / validation / test split
# ============================================================================
def _mouse_num(pid: str) -> int:
    """Turn a mouse ID like 'Mouse17' into the number 17 (used for sorting)."""
    return int(pid.replace("Mouse", ""))


def split_by_mouse(mouse_ids: List[str], num_train: int = 90, num_val: int = 15,
                   num_test: int = 25, seed: int = 42
                   ) -> Tuple[List[str], List[str], List[str]]:
    """
    Assign whole mice to the training, validation and test sets.

    Splitting by mouse (not by image) prevents data leakage: slices from the
    same heart look very similar, so if one heart were in both training and
    test, the test score would be unrealistically high.

    This reproduces the researchers' notebook (Pre-Clinical_CMR_Segmentation.ipynb,
    Step 5) exactly: sort the mouse IDs as text, set np.random.seed(42),
    shuffle, then take the first 90 / next 15 / last 25. With seed 42 this
    matches the released notebook lists (see PAPER_SPLIT below). This
    does not establish identity with the final paper training partition.

    Returns three lists of mouse IDs (train, val, test), each sorted by number.
    """
    # Remove duplicates and sort as text ('Mouse1', 'Mouse10', 'Mouse100', ...),
    # which is the order the notebook used before shuffling
    all_mice = sorted(set(mouse_ids))
    # Make sure the requested sizes add up to the number of mice available
    assert num_train + num_val + num_test == len(all_mice), \
        f"Split sizes ({num_train}+{num_val}+{num_test}) != number of mice ({len(all_mice)})"
    # Seed NumPy's legacy random generator (same call the notebook used)
    np.random.seed(seed)
    # Shuffle the mouse IDs in place
    shuffled = np.array(all_mice)
    np.random.shuffle(shuffled)
    # Cut the shuffled list into the three groups
    train = shuffled[:num_train].tolist()
    val = shuffled[num_train:num_train + num_val].tolist()
    test = shuffled[num_train + num_val:].tolist()
    # Return each group sorted by mouse number so it is easy to read
    return (sorted(train, key=_mouse_num), sorted(val, key=_mouse_num),
            sorted(test, key=_mouse_num))


# Released notebook split (90 / 15 / 25), copied from the output
# printed in their notebook. Used only to CHECK that split_by_mouse()
# reproduced it. Any mouse not listed here is a training mouse.
PAPER_SPLIT = {
    "val": ["Mouse5", "Mouse19", "Mouse25", "Mouse33", "Mouse34", "Mouse35", "Mouse39",
            "Mouse61", "Mouse63", "Mouse64", "Mouse80", "Mouse82", "Mouse92", "Mouse93",
            "Mouse128"],
    "test": ["Mouse6", "Mouse10", "Mouse12", "Mouse15", "Mouse29", "Mouse36", "Mouse46",
             "Mouse49", "Mouse56", "Mouse60", "Mouse65", "Mouse71", "Mouse74", "Mouse75",
             "Mouse78", "Mouse79", "Mouse87", "Mouse89", "Mouse91", "Mouse99", "Mouse100",
             "Mouse111", "Mouse117", "Mouse118", "Mouse125"],
}


def check_matches_paper(val: List[str], test: List[str]) -> bool:
    """
    Print YES/NO for whether our validation and test mice are exactly the
    same as the released notebook lists. This does not verify the final paper split.
    """
    # Compare as sets so the order of the mice doesn't matter
    same = set(val) == set(PAPER_SPLIT["val"]) and set(test) == set(PAPER_SPLIT["test"])
    print("Split matches the released UNet 3+ notebook split:", "YES" if same else "NO")
    return same


def save_split(path: str, train: List[str], val: List[str], test: List[str],
               seed: int) -> None:
    """Write the split to a .json file so it can be reused or shared."""
    with open(path, "w") as f:
        json.dump({"seed": seed, "train": train, "val": val, "test": test}, f, indent=2)
    print(f"Split saved: {path}")


def load_split(path: str) -> Tuple[List[str], List[str], List[str]]:
    """Read a split from a .json file written by save_split()."""
    with open(path) as f:
        d = json.load(f)
    print(f"Split loaded: {path}")
    return d["train"], d["val"], d["test"]


def select_by_mice(images: np.ndarray, masks: np.ndarray, filenames: List[str],
                   mice: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Keep only the 2D slices that belong to the given mice.

    Each slice has a name like 'Mouse13_sl05_fr12' (mouse, slice, frame);
    the part before the first underscore is the mouse ID.
    """
    # Convert to a set for fast "is this mouse in the list?" checks
    mice = set(mice)
    # Find the positions of every slice whose mouse ID is in the set
    idx = [i for i, fn in enumerate(filenames) if fn.split("_")[0] in mice]
    # Return the matching images, masks and filenames
    return images[idx], masks[idx], [filenames[i] for i in idx]


# ============================================================================
# 6. Summary statistics
# ============================================================================
def summarize_scores(rows: List[Dict], keys: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    """
    Compute the mean and standard deviation of each Dice score across the
    test mice.

    rows : one dictionary per mouse, e.g. {"mouse": "Mouse6", "average": 0.92, ...}
    keys : which scores to summarise (e.g. "diastolic_myocardium", "average")

    Returns {key: (mean, SD)}. Uses the sample SD (ddof=1), the usual choice
    when reporting mean ± SD across subjects.
    """
    out = {}
    for k in keys:
        # Collect this score from every mouse, skipping missing or NaN values
        vals = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])], dtype=float)
        if len(vals):
            # SD needs at least 2 values; with only 1 mouse report SD as 0
            out[k] = (float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
    return out
