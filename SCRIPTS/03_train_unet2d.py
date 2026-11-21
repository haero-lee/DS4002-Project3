"""
===============================================================================
03_train_unet2d.py
===============================================================================
PURPOSE
    Train a plain 2D U-Net to segment the left ventricle (LV) myocardium and
    blood pool in mouse cine short-axis cardiac MRI, then test it on mice the
    model has never seen. Test scoring uses per-mouse Dice at end-diastole
    (ED) and end-systole (ES), using the released utility code. Comparison
    with published UNet 3+ summaries is descriptive; protocol equivalence
    and identity with the final paper partition are not established.

PROJECT CONTEXT
    DS 4002 Project 3: 2D U-Net vs. UNet 3+ for mouse LV segmentation.
    Data and UNet 3+ code: https://github.com/mrphys/Open-Source_Pre-Clinical_Segmentation
    Goal (finish line): mean test Dice >= 0.90.

REQUIRED FILES (scripts in SCRIPTS/, commands below run from SCRIPTS/)
    03_train_unet2d.py   <- this script (run this one)
    04_unet2d_utils.py   <- model, loss, metrics, mouse-level split
    02_utility.py        <- researchers' helpers (loading, augmentation, Dice evaluation)
    ../DATA/WBH5Data/ <- folder holding Mouse1.h5 ... Mouse130.h5
                         (or point to another folder with --data_dir)

DATA
    130 mice, one HDF5 (.h5) file each. Each file contains:
        Images           256 x 256 x slices (8-11) x timeframes (17-36)
        Masks            same size; 0 = background, 1 = myocardium, 2 = blood pool
        Frames_with_ROI  which timeframes have expert labels (ED, ES, sometimes inflow)
    Only labeled timeframes are used for training.

SPLIT
    90 / 15 / 25 mice (train / validation / test), seed 42. This matches the
    released notebook assignments. Identity with the final paper partition
    is not independently confirmed. The script checks the notebook lists.

REQUIREMENTS
    Python 3.9+ (3.11 recommended), an NVIDIA GPU is strongly recommended.
    pip install "tensorflow[and-cuda]" h5py albumentations opencv-python scikit-image matplotlib numpy

HOW TO RUN (from a terminal, inside the folder above)
    conda activate ml_env

    Quick test (1 epoch, checks everything works):
        python 03_train_unet2d.py --gpu 0 --epochs 1 --data_dir ../DATA/WBH5Data --out_dir ../OUTPUT/test_run

    Full training in the background (keeps running after you log out):
        nohup python 03_train_unet2d.py --gpu 0 --epochs 300 --data_dir ../DATA/WBH5Data --out_dir ../OUTPUT > train_unet2d.log 2>&1 &

    Check progress:
        tail -f train_unet2d.log        (Ctrl+C stops watching, not training)

    Reuse a saved split instead of regenerating it:
        python 03_train_unet2d.py --gpu 0 --data_dir ../DATA/WBH5Data --out_dir ../OUTPUT --split_json ../OUTPUT/UNet2D_2026-11-16_17-16-24/split.json

COMMAND-LINE OPTIONS (all optional; defaults in brackets)
    --data_dir [WBH5Data]     folder with the .h5 files
    --out_dir [runs]          where results are saved
    --gpu [all visible]       which GPU to use, e.g. 0 or 1
    --epochs [100]            maximum number of training epochs
    --patience [10]           stop if validation Dice doesn't improve for this many epochs
    --batch_size [16]  --lr [1e-4]  --seed [42]  --base_filters [32]  --dropout [0.0]
    --num_train/--num_val/--num_test [90/15/25]
    --split_json              reuse an existing split file
    --mixed_precision         use 16-bit math (faster on newer GPUs)
    --no-island_removal       turn off the clean-up step before scoring
    --target_dice [0.90]      finish-line Dice

OUTPUTS  ->  runs/UNet2D_<date>_<time>/
    best_model.keras              model from the epoch with the best validation Dice
    final_model.keras             model at the end of training (best weights restored)
    split.json, config.json       which mice were used and the training settings
    training_log.csv, history.json  loss and Dice for every epoch
    loss_plot.png, dice_combined.png  training curves
    eval/test_dice_per_mouse.csv  Dice for each test mouse
    eval/test_summary.txt         mean ± SD Dice and whether the finish line was met

NOTES
    - Each run creates a NEW timestamped folder; old runs are never overwritten.
    - The test mice are only used once, at the very end, after training is done.
    - Training stops early when validation Dice stops improving, and the best
      model (not the last) is used for testing.

AUTHOR / DATE
    Haero Lee (DS 4002, Project 3), November 2026
===============================================================================
"""

# Only these two modules are imported before the arguments are read, because
# the GPU choice (--gpu) has to be set BEFORE TensorFlow is imported.
import argparse   # reads options typed on the command line (e.g. --epochs 300)
import os         # sets environment variables and builds file paths


def parse_args():
    """Define every command-line option, its default, and its help text."""
    p = argparse.ArgumentParser(description="Train 2D U-Net on mouse cine SAX MRI")

    # ---- Data and output locations ----
    p.add_argument("--data_dir", default="WBH5Data",
                   help="Folder with Mouse1.h5 ... Mouse130.h5 (default: WBH5Data)")
    # Number of mice in each set (must add up to 130)
    p.add_argument("--num_train", type=int, default=90)
    p.add_argument("--num_val", type=int, default=15)
    p.add_argument("--num_test", type=int, default=25)
    p.add_argument("--out_dir", default="runs", help="Base folder for run outputs")
    p.add_argument("--split_json", default=None,
                   help="Optional existing split.json to reuse instead of making a new split")

    # ---- Hardware ----
    p.add_argument("--gpu", default=None, help="GPU index to use, e.g. '0' or '1'")
    p.add_argument("--mixed_precision", action="store_true")

    # ---- Training settings ----
    p.add_argument("--seed", type=int, default=42)          # random seed (split + training)
    p.add_argument("--batch_size", type=int, default=16)    # images per training step
    p.add_argument("--epochs", type=int, default=100)       # maximum passes over the data
    p.add_argument("--lr", type=float, default=1e-4)        # starting learning rate (Adam)
    p.add_argument("--patience", type=int, default=10, help="Early-stopping patience (epochs)")
    p.add_argument("--base_filters", type=int, default=32)  # filters at the first U-Net level
    p.add_argument("--dropout", type=float, default=0.0)    # dropout at the bottleneck (0 = off)

    # ---- Evaluation settings ----
    # BooleanOptionalAction creates both --island_removal and --no-island_removal
    p.add_argument("--island_removal", action=argparse.BooleanOptionalAction, default=True,
                   help="Apply utility.island_removal before scoring, as the UNet 3+ notebook "
                        "does (default: on; use --no-island_removal to turn off)")
    p.add_argument("--target_dice", type=float, default=0.90, help="Finish-line mean Dice")
    return p.parse_args()


# Read the command-line options right away (needed before importing TensorFlow)
args = parse_args()

# Choose which GPU TensorFlow can see. This MUST happen before TensorFlow is
# imported, otherwise TensorFlow will already have claimed all GPUs.
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
# Hide TensorFlow's routine info/warning messages so the log stays readable
# (errors are still shown)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---- Remaining imports (safe to load now that the GPU is chosen) ----
import csv    # write the per-mouse test Dice table
import math   # math.ceil for the number of training steps per epoch

import matplotlib
# "Agg" draws plots straight to image files without opening a window.
# Needed on a GPU server that has no screen attached.
matplotlib.use("Agg")

import numpy as np                                  # array handling
import tensorflow as tf                             # model training
from tensorflow.keras.utils import to_categorical   # turns label maps into one-hot form

from importlib import import_module
# Numbered filenames must be loaded through importlib, not import statements.
U = import_module('02_utility')
UU = import_module('04_unet2d_utils')


def main():
    # ======================================================================
    # STEP 1: Setup
    # ======================================================================
    # Seed all random number generators so the run can be repeated
    UU.set_seeds(args.seed)
    # Configure the GPU (memory growth, optional mixed precision)
    UU.setup_gpu(mixed_precision=args.mixed_precision)

    # Create a new folder for this run, e.g. runs/UNet2D_2026-11-16_17-20-45/
    # (also creates the eval/ subfolder)
    run_dir = U.make_run_dir(base=args.out_dir, model_name="UNet2D")
    print(f"Run directory: {run_dir}")

    # ======================================================================
    # STEP 2: Load the data
    # ======================================================================
    print("\nLoading data (labeled frames only, from Frames_with_ROI)...")
    # Reads every Mouse*.h5 file and returns:
    #   images, masks     -> every labeled 2D slice and its mask, (N, 256, 256)
    #   filenames         -> one name per slice, e.g. 'Mouse13_sl05_fr12'
    #   relevant_frames   -> {mouse: list of labeled timeframes}
    #   all_volumes       -> {mouse: full 4D cine images}, used for testing
    #   all_masks         -> {mouse: full 4D masks}, used for testing
    #   masks_check       -> any masks with unexpected label values
    # It also works out each mouse's ED and ES frame (stored in U.ed_index_map
    # and U.es_index_map) for scoring later.
    (images, masks, filenames, relevant_frames,
     all_volumes, all_masks, masks_check) = U.load_roi_data_from_h5_folder(args.data_dir)
    # Print a summary: number of mice, labeled images, and a mask value check
    U.summarise_mice_and_masks(relevant_frames, masks_check)

    # ======================================================================
    # STEP 3: Split mice into training / validation / test
    # ======================================================================
    all_mice = list(relevant_frames.keys())
    if args.split_json:
        # Reuse a split saved earlier
        train_mice, val_mice, test_mice = UU.load_split(args.split_json)
    else:
        # Make the 90/15/25 split (released notebook assignments with seed 42)
        train_mice, val_mice, test_mice = UU.split_by_mouse(
            all_mice, num_train=args.num_train, num_val=args.num_val,
            num_test=args.num_test, seed=args.seed)
    # Print YES/NO for whether the split matches the released notebook lists
    UU.check_matches_paper(val_mice, test_mice)
    # Save the split inside this run's folder for the record
    UU.save_split(os.path.join(run_dir, "split.json"), train_mice, val_mice, test_mice, args.seed)

    # Safety check: stop immediately if any mouse appears in two sets
    assert not (set(train_mice) & set(val_mice) or set(train_mice) & set(test_mice)
                or set(val_mice) & set(test_mice)), "Mouse appears in more than one split!"

    # Pull out the 2D slices that belong to each group of mice
    tr_x, tr_y, _ = UU.select_by_mice(images, masks, filenames, train_mice)
    va_x, va_y, _ = UU.select_by_mice(images, masks, filenames, val_mice)
    te_x, te_y, _ = UU.select_by_mice(images, masks, filenames, test_mice)
    # Print how many mice and images ended up in each set, and which mice
    U.train_val_test_dis(all_mice, train_mice, val_mice, test_mice,
                         tr_x, tr_y, va_x, va_y, te_x, te_y)

    # ======================================================================
    # STEP 4: Prepare the images for the model
    # ======================================================================
    # Scale each image to the range [0, 1] (min-max per image), the same way
    # the researchers did. Raw MRI intensities vary a lot between scans.
    tr_x = U.normalise_images(tr_x)
    # Validation images also get a channel dimension: (N, 256, 256) -> (N, 256, 256, 1)
    va_x = U.normalise_images(va_x)[..., None]
    # Turn validation label maps into one-hot form: (N, 256, 256) -> (N, 256, 256, 3)
    # so they match the model's 3-channel output
    va_y = to_categorical(va_y.astype(np.int32), num_classes=UU.NUM_CLASSES).astype(np.float32)

    # Training data comes from a generator that, every batch, picks random
    # slices, applies random augmentation (flips, rotation, brightness), and
    # converts masks to one-hot form. Validation data is NOT augmented.
    train_gen = U.data_generator_with_augmentation(
        tr_x, tr_y.astype(np.int32), args.batch_size, UU.get_augmentation())
    # Number of batches per epoch, so each epoch sees about as many images as
    # the training set contains
    steps_per_epoch = math.ceil(len(tr_x) / args.batch_size)

    # ======================================================================
    # STEP 5: Build the model
    # ======================================================================
    # Filter widths for the 5 U-Net levels: 32, 64, 128, 256, 512 by default
    filters = [args.base_filters * 2 ** i for i in range(5)]
    model = UU.build_unet2d(input_shape=(256, 256, 1), filters=filters,
                            num_classes=UU.NUM_CLASSES, dropout=args.dropout)
    # Connect the model to its optimiser, loss and metrics:
    #   - Adam optimiser with the chosen learning rate
    #   - Dice + cross-entropy loss
    #   - Dice for myocardium, blood pool, and their average, reported every epoch
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss=UU.dice_ce_loss,
                  metrics=[UU.dice_myo, UU.dice_blood, UU.dice_fg])
    # Print every layer and its output shape (useful for checking the model)
    model.summary(print_fn=lambda s: print(s))
    print(f"Trainable parameters: {model.count_params():,}")

    # Record the settings of this run in config.json for reproducibility
    cfg = U.RunConfig(
        model_name="UNet2D", batch_size=args.batch_size, epochs=args.epochs,
        input_shape=(256, 256, 1), optimizer=f"Adam(lr={args.lr})",
        loss="soft Dice (fg) + categorical CE", metrics=["dice_myo", "dice_blood", "dice_fg"],
        train_n=len(tr_x), val_n=len(va_x), test_n=len(te_x),
        train_mice=train_mice, val_mice=val_mice, test_mice=test_mice)
    U.dump_config_json(run_dir, cfg)

    # ======================================================================
    # STEP 6: Callbacks (actions Keras runs automatically after each epoch)
    # ======================================================================
    best_path = os.path.join(run_dir, "best_model.keras")
    callbacks = [
        # Save the model whenever validation foreground Dice reaches a new high
        tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_dice_fg", mode="max",
                                           save_best_only=True, verbose=1),
        # Stop training if validation Dice hasn't improved for `patience` epochs,
        # then roll the model back to its best epoch
        tf.keras.callbacks.EarlyStopping(monitor="val_dice_fg", mode="max",
                                         patience=args.patience, restore_best_weights=True,
                                         verbose=1),
        # Halve the learning rate if validation loss hasn't improved for 5 epochs,
        # which lets the model make finer adjustments (never below 1e-6)
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                                             min_lr=1e-6, verbose=1),
        # Write loss, Dice and learning rate for every epoch to training_log.csv
        tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "training_log.csv")),
    ]

    # ======================================================================
    # STEP 7: Train
    # ======================================================================
    history = model.fit(train_gen,                        # augmented training batches
                        steps_per_epoch=steps_per_epoch,  # batches per epoch
                        validation_data=(va_x, va_y),     # checked at the end of each epoch
                        validation_batch_size=args.batch_size,
                        epochs=args.epochs,               # upper limit; early stopping may end sooner
                        callbacks=callbacks,
                        verbose=2)                        # one line of output per epoch

    # Save the model after training (EarlyStopping already restored the best weights)
    model.save(os.path.join(run_dir, "final_model.keras"))
    # Save the per-epoch history as JSON and draw the loss and Dice curves
    U.export_history_json(history, run_dir)
    U.save_loss_and_dice_plots(history, out_dir=run_dir)

    # ======================================================================
    # STEP 8: Evaluate on the held-out test mice
    # ======================================================================
    # Using the released utility: for each test mouse, Dice is computed on
    # the whole stack of short-axis slices at ED and at ES, for myocardium and
    # blood pool separately.
    print("\nEvaluating on held-out test mice (ED/ES per-mouse Dice)...")
    # Build each test mouse's full cine (all slices, all timeframes), normalised
    # the same way as the training images
    imgs4d, frame_ids = U.build_fullcine_for(test_mice, source_volumes=all_volumes)
    # Run the model on every slice of every timeframe; keep the most likely
    # label (0/1/2) for each pixel
    preds4d = U.predict_per_patient_in_memory(model, imgs4d, verbose=0)

    # Optional clean-up (on by default, as in the researchers' notebook):
    # remove small or far-away blobs of predicted myocardium/blood pool that
    # are clearly not part of the heart
    if args.island_removal:
        for pid, pr in preds4d.items():
            for f in range(pr.shape[3]):
                pr[..., f] = U.island_removal(pr[..., f])

    # The five scores reported for each mouse
    keys = ["diastolic_myocardium", "diastolic_blood_pool",
            "systolic_myocardium", "systolic_blood_pool", "average"]
    rows = []
    for pid in test_mice:
        # Compute (and print) this mouse's ED/ES Dice scores
        scores = U.print_patient_ed_es_dice_summary(
            pid=pid, images_4d=imgs4d[pid], preds_4d=preds4d[pid],
            masks_4d=all_masks[pid], frame_id_list=frame_ids[pid],
            ed_map=U.ed_index_map, es_map=U.es_index_map)
        # Keep one row per mouse; use NaN if a score couldn't be computed
        rows.append({"mouse": pid, **{k: scores.get(k, float("nan")) for k in keys}})
        print()

    # Save every mouse's scores to eval/test_dice_per_mouse.csv
    csv_path = os.path.join(run_dir, "eval", "test_dice_per_mouse.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mouse"] + keys)
        w.writeheader()
        w.writerows(rows)

    # ======================================================================
    # STEP 9: Summarise and check the finish line
    # ======================================================================
    # Mean ± SD of each score across the test mice
    summary = UU.summarize_scores(rows, keys)
    # Build the text of the summary report
    lines = ["=" * 44, f"Test set summary (n = {len(test_mice)} mice), mean ± SD", "=" * 44]
    for k, (m, s) in summary.items():
        lines.append(f"{k:<24} {m:.4f} ± {s:.4f}")
    # Compare the overall mean Dice with the finish line (0.90 by default)
    mean_dice = summary.get("average", (float("nan"), 0))[0]
    status = "MET" if mean_dice >= args.target_dice else "NOT MET"
    lines.append(f"\nFinish line (mean Dice >= {args.target_dice:.2f}): {status}")
    report = "\n".join(lines)
    # Print the report to the log and save it to eval/test_summary.txt
    print(report)
    with open(os.path.join(run_dir, "eval", "test_summary.txt"), "w") as f:
        f.write(report + "\n")

    print(f"\nDone. Outputs saved in: {run_dir}")


# Run main() only when this file is executed directly (python 03_train_unet2d.py),
# not when it is imported from another script
if __name__ == "__main__":
    main()
