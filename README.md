# DS4002-Project3

This project trains a standard 2D U-Net to segment left ventricular myocardium and blood pool in mouse cardiac MRI and compares its Dice scores descriptively with published UNet 3+ results. The target is mean foreground Dice of at least 0.90 on held-out mice. We will report the validation-selected model even if it misses that target. We will not retrain UNet 3+.

## Section 1: Software and platform

The existing EDA notebook uses Python, Jupyter/Google Colab, NumPy, pandas, h5py, Matplotlib, and SciPy. For exploration, install these in your Python environment:

```bash
python -m pip install numpy pandas h5py matplotlib scipy notebook
```

There is currently no `requirements.txt`. The actual training environment, package versions, platform, and commands will be documented when the training scripts are uploaded. The earlier README reported preprocessing on Windows with Python 3.12 and pandas 3.0.1; a complete pipeline has not been verified across platforms.

## Section 2: A Map of our documentation

```text
DS4002-Project3/
├── README.md                    Project overview and workflow
├── LICENSE                      Project license
├── DATA/
│   ├── README.MD                 Data provenance and dictionary
│   └── published_unet3plus.csv    Published benchmark scores and citations
└── SCRIPTS/
    └── EDA_py.ipynb              Exploratory data analysis
```

The 2D U-Net has been trained, but its training scripts, actual mouse assignments, checkpoint details, and results have not yet been uploaded to this checkout. There is no `OUTPUT/` folder or completed comparison here yet.

## Section 3: Instructions for Reproducing Our Results

### Published benchmark available now

[DATA/published_unet3plus.csv](DATA/published_unet3plus.csv) contains the internal 9.4T benchmark from the final paper's Table 1 [1]. The numbers were transcribed from the paper; they were not calculated by this project. Each row includes the mean Dice, standard deviation, test cohort size, DOI, table reference, source URL, and post-processing description. `result_origin=published_not_reproduced` identifies their origin.

| Metric | Published Dice mean | Published SD |
| --- | ---: | ---: |
| Blood pool, ED | 0.95 | 0.02 |
| Myocardium, ED | 0.89 | 0.03 |
| Blood pool, ES | 0.93 | 0.03 |
| Myocardium, ES | 0.92 | 0.02 |
| Overall | 0.92 | 0.02 |

These results concern 25 internal test mice. ED means end-diastole; ES means end-systole. The paper used 90/15/25 training/validation/test mice, evaluated its internal test mice at ED/ES, and applied connected-component post-processing [1]. The 0.95 score is blood-pool ED Dice, not overall Dice. To verify the CSV, compare its five rows with the internal 9.4T column of Table 1 at the linked source. No script is needed to obtain these published summary scores.

### Comparison after the U-Net scripts and results are uploaded

1. Document the mouse assignments actually used for the completed U-Net training. Do not replace or regenerate that split after training. Check that training, validation, and test mice do not overlap.
2. Inspect the uploaded evaluation code and results before adding another evaluator. Record the data version, software environment, selected checkpoint, test cohort size, frame selection, Dice aggregation, empty-class handling, and any post-processing. If predictions need evaluation, verify mouse/frame/slice alignment, shapes, and labels. Use `Frames_with_ROI` to identify annotated frames and retain valid background-only slices within them.
3. Confirm whether the reported scores match the planned evaluation: pool class pixel counts across each mouse's selected slices and frames, calculate myocardium and blood-pool Dice, average the two for that mouse, then report means and standard deviations across equally weighted test mice. Document any differences from the MI2 plan.
4. Create a comparison table and figure from the project's measured scores and the published CSV, clearly distinguishing their origins. Save measured summaries and presentation artifacts in `OUTPUT/`. Include the planned best/worst overlays and slice-position analysis where supported by the uploaded data.
5. Add exact commands to regenerate the measured results and comparison using the uploaded code, actual split, data version, dependencies, and checkpoint identity.

The comparison will be descriptive. Differences in mouse assignments, preprocessing, training, evaluated phases, post-processing, or aggregation may affect scores. Do not assume the paper's overall Dice matches our aggregation exactly. Published summary means and SDs alone do not support a paired per-mouse significance test. Complete reproduction commands remain pending the model scripts and results.

## References

[1] W. Shah et al., “Open-source pre-clinical image segmentation: mouse cardiac magnetic resonance imaging datasets with a deep learning segmentation framework,” *J. Cardiovasc. Magn. Reson.*, vol. 28, no. 1, Art. no. 102706, 2026, doi: [10.1016/j.jocmr.2026.102706](https://doi.org/10.1016/j.jocmr.2026.102706). [Final paper and Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC13237537/#tbl1).

[2] mrphys, “Open-Source_Pre-Clinical_Segmentation,” GitHub. [Source repository](https://github.com/mrphys/Open-Source_Pre-Clinical_Segmentation). Accessed: Nov. 17, 2026.
