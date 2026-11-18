# DS4002-Project3

This project trains a standard 2D U-Net to segment the left ventricular (LV) myocardium and blood pool in mouse cardiac MRI, then compares its Dice scores with the published UNet 3+ results [1]. Our target was a mean Dice of at least 0.90 on 25 held-out test mice.

**Result:** the 2D U-Net reached a mean test Dice of **0.902 ± 0.049**, meeting the 0.90 target. It matched the UNet 3+ at end-diastole but scored lower at end-systole, especially for the myocardium. We did not retrain the UNet 3+. Its scores are taken from the published paper.

## Section 1: Software and Platform

**Software**
- Python 3.11 (model training and evaluation)
- Jupyter Notebook / Google Colab (exploratory data analysis)

**Packages**

| Package | Used for |
|---|---|
| `tensorflow` (with GPU support) | Building and training the 2D U-Net |
| `h5py` | Reading the `.h5` MRI data files |
| `numpy` | Array operations |
| `albumentations`, `opencv-python` | Data augmentation during training |
| `scikit-image` | Post-processing (island removal) and cardiac phase detection |
| `matplotlib` | Training curves and figures |
| `pandas`, `scipy` | Exploratory data analysis |

Install everything with:

```bash
pip install "tensorflow[and-cuda]" h5py numpy albumentations opencv-python scikit-image matplotlib pandas scipy notebook
```

**Platform**
- Model training and evaluation: Linux workstation with an NVIDIA GPU (Epstein Lab, UVA)
- Exploratory data analysis: Windows with Python 3.12 and Google Colab

## Section 2: A Map of Our Documentation

```text
DS4002-Project3/
├── README.md                        Project overview and reproduction instructions (this file)
├── LICENSE                          Project license
├── DATA/
│   ├── README.MD                    Data provenance, license, data dictionary, and explanatory plots
│   ├── published_unet3plus.csv      Published UNet 3+ benchmark scores with citations
│   └── split_90_15_25.json          Mice assigned to the training, validation, and test sets
├── SCRIPTS/
│   ├── EDA_py.ipynb                 Exploratory data analysis
│   ├── train_unet2d.py              Main script: loads data, trains the 2D U-Net, evaluates on test mice
│   ├── unet2d_utils.py              2D U-Net model, loss function, metrics, and mouse-level data split
│   └── utility.py                   Data loading, augmentation, and Dice evaluation from the UNet 3+ authors (GPL-3.0) [2]
└── OUTPUT/
    ├── test_summary.txt             Mean ± SD test Dice and whether the 0.90 target was met
    ├── test_dice_per_mouse.csv      Dice scores for each of the 25 test mice
    ├── training_log.csv             Loss and Dice for every training epoch
    ├── loss_plot.png                Training and validation loss curves
    ├── dice_combined.png            Training and validation Dice curves
    └── config.json                  Training settings used for the reported run
```

**Not included in this repository:**
- **The MRI dataset (about 5 GB)** is too large for GitHub. Section 3, Step 2 explains how to download it.
- **The trained model weights (`best_model.keras`)** are also too large for GitHub. Running the steps in Section 3 regenerates them.

## Section 3: Instructions for Reproducing Our Results

These steps reproduce our full analysis pipeline, from downloading the data to the final figures. Every command is run in a terminal.

| Pipeline stage | Where it happens |
|---|---|
| 1. Data extraction | `train_unet2d.py` (Step 5 below) |
| 2. Data splitting | `train_unet2d.py` (Step 5) |
| 3. Preprocessing and augmentation | `train_unet2d.py` (Step 5) |
| 4. Model training | `train_unet2d.py` (Step 5) |
| 5. Evaluation on test mice | `train_unet2d.py` (Step 5) and `analyze_results.py` (Step 7) |
| 6. Benchmark comparison | `analyze_results.py` (Step 7) |

### Requirements

- **Operating system:** Linux is recommended (we used a Linux lab workstation). macOS or Windows (through WSL2) can run the code, but training without an NVIDIA GPU is very slow.
- **Hardware:** an NVIDIA GPU is strongly recommended. Training took a few hours on our lab GPU. Step 7 can run on any computer.
- **Disk space:** about 6 GB.
- **Software:** [Anaconda or Miniconda](https://docs.conda.io/en/latest/miniconda.html) and Git.

### Step 1: Clone this repository

```bash
git clone https://github.com/<your-username>/DS4002-Project3.git
cd DS4002-Project3
```

### Step 2: Download the data

The data comes from the mrphys Open-Source Pre-Clinical Segmentation repository [2]. It contains cine short-axis MRI scans of 130 mice, one `.h5` file per mouse (`Mouse1.h5` to `Mouse130.h5`). Download it and move the data into `DATA/WBH5Data`:

```bash
git clone https://github.com/mrphys/Open-Source_Pre-Clinical_Segmentation.git
mv Open-Source_Pre-Clinical_Segmentation/Data DATA/WBH5Data
ls DATA/WBH5Data | wc -l     # should print 130
```

### Step 3: Set up the Python environment

```bash
conda create -n ml_env python=3.11 -y
conda activate ml_env
pip install "tensorflow[and-cuda]" h5py numpy pandas scipy albumentations opencv-python scikit-image matplotlib
```

Check that the packages installed and that TensorFlow can see the GPU:

```bash
python -c "import tensorflow as tf, h5py, albumentations, cv2, skimage; print('All imported'); print(tf.config.list_physical_devices('GPU'))"
```

You should see `All imported` followed by a GPU, such as `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`. An empty list (`[]`) means TensorFlow cannot see the GPU. Training will still run, but much more slowly.

### Step 4: Run a quick test (1 epoch)

Move into `SCRIPTS` and train for one epoch to confirm everything works. This takes a few minutes.

```bash
cd SCRIPTS
python train_unet2d.py --gpu 0 --epochs 1 --data_dir ../DATA/WBH5Data --out_dir ../OUTPUT/test_run
```

Check that the output includes `Split matches the UNet 3+ paper split: YES` and `Number of training mice: 90`, `validation mice: 15`, `test mice: 25`, followed by Dice scores for 25 test mice. These scores are not meaningful. Delete `OUTPUT/test_run` afterwards.

### Step 5: Train and evaluate the model

Still in `SCRIPTS`, run the full training. `nohup` and the `&` at the end keep it running in the background after you close the terminal.

```bash
nohup python train_unet2d.py --gpu 0 --epochs 300 --data_dir ../DATA/WBH5Data --out_dir ../OUTPUT > train_unet2d.log 2>&1 &
```

To check progress (press `Ctrl+C` to stop watching; training continues):

```bash
tail -f train_unet2d.log
```

This one command runs pipeline stages 1–5:

1. **Data extraction:** loads `Images`, `Masks`, and `Frames_with_ROI` from each mouse's `.h5` file. Only labeled timeframes are kept, and each is split into 2D 256 × 256 slices tagged with mouse ID, slice, and frame. Unlabeled frames are excluded, not imputed. Each mouse's end-diastolic (ED) frame is its first labeled frame, and its end-systolic (ES) frame is the labeled frame with the smallest blood pool.
2. **Data splitting:** whole mice (not slices) are split 90 / 15 / 25 into training / validation / test with seed 42. This is the same split used for the published UNet 3+ [1], [2], so both models are tested on the same 25 mice. Splitting by mouse prevents leakage between sets. The split is saved as `split.json`, and the test mice are not used until final evaluation.
3. **Preprocessing:** each slice is min–max scaled to [0, 1] and kept at 256 × 256. Masks are one-hot encoded into 3 channels (background, myocardium, blood pool). Training slices only are augmented with random horizontal and vertical flips, rotation (±15°), and brightness/contrast changes (±10%).
4. **Model training:**

   | Setting | Value |
   |---|---|
   | Model | 2D U-Net [3], 4 downsampling levels, 32 base filters doubling at each level (32–512), batch normalization, 3-class softmax output |
   | Loss | Soft Dice (myocardium + blood pool) + categorical cross-entropy |
   | Optimizer | Adam, learning rate 1e-4, halved when validation loss stops improving for 5 epochs |
   | Batch size | 16 |
   | Epochs | Up to 300; stops early after 10 epochs without improvement in validation Dice; the best checkpoint is kept |

5. **Evaluation:** the best model predicts every slice of each test mouse's full cine, assigning each pixel its most likely class. Small stray blobs are then removed (island removal), as in the UNet 3+ paper. For each mouse, Dice is computed over the whole stack of slices at ED and at ES, separately for the myocardium and blood pool. The overall score is the mean of these four values. Results are reported as mean ± SD across the 25 test mice.

Training is finished when the log ends with `Done. Outputs saved in: ../OUTPUT/UNet2D_<date>_<time>`.

### Step 6: View the training results

The training outputs are in a new folder, `OUTPUT/UNet2D_<date>_<time>/`:

| File | Contents |
|---|---|
| `eval/test_summary.txt` | Mean ± SD test Dice and whether the 0.90 target was met |
| `eval/test_dice_per_mouse.csv` | Dice scores for each of the 25 test mice |
| `training_log.csv` | Loss and Dice for every epoch |
| `loss_plot.png`, `dice_combined.png` | Training and validation curves |
| `best_model.keras` | Trained model from the epoch with the highest validation Dice |
| `config.json`, `split.json` | Training settings and the mice in each set |

```bash
cat ../OUTPUT/UNet2D_<date>_<time>/eval/test_summary.txt
```

### Step 7: Analyze the results and compare with UNet 3+

From inside `SCRIPTS`, replacing `<date>_<time>` with your run folder's name:

```bash
python analyze_results.py --run_dir ../OUTPUT/UNet2D_<date>_<time> --overlays --gpu 0
```

This runs pipeline stages 5 and 6. It:
- computes summary statistics (mean, SD, median, range) and the number of mice that reached Dice ≥ 0.90
- identifies the best, median, and worst test mice, and repeats the summary without the lowest-scoring mouse as a sensitivity check
- tests whether Dice differs between ED and ES for each structure, using a paired, two-sided Wilcoxon signed-rank test (α = 0.05)
- compares our results with the published UNet 3+ scores [1] (also in [DATA/published_unet3plus.csv](DATA/published_unet3plus.csv))
- with `--overlays`, saves images of the model's segmentation next to the expert segmentation at ED and ES for the best, median, and worst mice

| Output (in `OUTPUT/`) | Contents |
|---|---|
| `analysis_summary.txt` | All statistics in plain text |
| `comparison_table.csv` | 2D U-Net vs. UNet 3+, by metric |
| `fig_dice_comparison.png` | Mean ± SD Dice for both models |
| `fig_dice_per_mouse.png` | Dice for every test mouse |
| `overlays/` | Predicted vs. expert segmentations for the best, median, and worst mice |

Without `--overlays`, this step needs only the per-mouse CSV, not the model or data, and takes a few seconds on any computer.

### Expected results

Our run stopped early at epoch 71, and the best model came from epoch 61. Dice is mean ± SD across the same 25 test mice:

| Metric | 2D U-Net (ours, measured) | UNet 3+ (published [1]) |
|---|---:|---:|
| Blood pool, ED | 0.946 ± 0.021 | 0.95 ± 0.02 |
| Myocardium, ED | 0.881 ± 0.045 | 0.89 ± 0.03 |
| Blood pool, ES | 0.899 ± 0.091 | 0.93 ± 0.03 |
| Myocardium, ES | 0.880 ± 0.089 | 0.92 ± 0.02 |
| **Overall** | **0.902 ± 0.049** | **0.92 ± 0.02** |

- **Finish line met:** the mean overall Dice of 0.902 reached the ≥ 0.90 target. Individually, 17 of 25 mice reached 0.90.
- **ED vs. ES:** blood pool Dice was significantly lower at ES than at ED (p = 0.0001). Myocardium Dice did not differ between the two (p = 0.067).
- **Comparison with UNet 3+:** the two models were nearly equal at ED (differences of 0.01 or less). The 2D U-Net scored lower at ES: 0.03 lower for the blood pool and 0.04 lower for the myocardium.
- **Best and worst mice:** the best was Mouse100 (overall 0.947), the median was Mouse78 (0.911), and the worst was Mouse91 (0.705, driven by ES Dice of 0.50–0.56). Without Mouse91, overall Dice is 0.910 ± 0.027. The main results include all 25 mice.

**Limitations:** the UNet 3+ scores are published summary values, not our own runs. Without its per-mouse scores, the models can only be compared descriptively, and no paired significance test between them is possible. Apart from the architecture, the models also differ in loss function, learning-rate schedule, and augmentation.

**Note on reproducibility:** all random seeds are fixed (seed 42), and the data split is fully deterministic. However, TensorFlow GPU computations are not perfectly deterministic, so a rerun may stop at a slightly different epoch and give slightly different Dice scores. The overall conclusions should not change.

### Deviations from the MI2 plan

| MI2 plan | What we did | Reason |
|---|---|---|
| 70 / 15 / 15 split (91 / 19 / 20 mice) | 90 / 15 / 25 split | Matches the UNet 3+ paper's split exactly, so both models are tested on the same 25 mice |
| Clip intensities to the 1st–99th percentile, then rescale | Per-slice min–max scaling to [0, 1] | Matches the original authors' preprocessing (`utility.py`) |
| Up to 100 epochs | Up to 300 epochs | Gives early stopping more room; training stopped at epoch 71 |
| Dice over all of a mouse's labeled slices | Dice over all slices at ED and ES, with island removal | Matches the UNet 3+ paper's evaluation, so the scores are directly comparable |
| Limited tuning of learning rate and base filters | No tuning | The first configuration met the 0.90 target |
| Optional Wilcoxon test against the released UNet 3+ weights | Not performed; compared against published scores instead, and added a within-model ED vs. ES Wilcoxon test | Dropped to keep the project in scope |
| Train on Google Colab or UVA HPC | Trained on an Epstein Lab Linux GPU workstation | A lab GPU was available |

### Troubleshooting

- **`ModuleNotFoundError`:** activate the environment (`conda activate ml_env`) and rerun the `pip install` command from Step 3.
- **`FileNotFoundError`, or 0 mice loaded:** check that the data is in `DATA/WBH5Data` and that you are running commands from inside `SCRIPTS`.
- **Out of GPU memory:** add `--batch_size 8` to the training command.
- **GPU 0 is busy:** run `nvidia-smi` to see which GPUs are free, then use `--gpu 1`.
- **Stopping a running job:** find its process ID with `ps aux | grep train_unet2d.py`, then run `kill <process_id>`.
