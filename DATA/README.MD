# Metadata and Data Documentation

## Data Summary

Our dataset consists of short-axis cine cardiac MRI images and expert segmentation masks from 130 mice. Each mouse has one HDF5 file containing MRI images, corresponding segmentation masks, and an indicator identifying annotated cardiac timeframes [3]. Saved outputs in our EDA notebook report 130 files, image dimensions of 256 x 256 pixels, 8-11 slices per mouse, and 17-36 cardiac timeframes [8]. Each model input will be a two-dimensional MRI slice, and its target mask will identify background, left ventricular myocardium, and blood pool. We will divide the dataset by mouse to prevent correlated images from appearing in the training, validation, and testing subsets.

## Provenance

The dataset was collected and released by Shah et al. for their study of automated mouse cardiac MRI segmentation [2]. It is publicly distributed through the mrphys/Open-Source_Pre-Clinical_Segmentation GitHub repository, which also provides the U-Net 3+ implementation and trained weights [3]. We will obtain the source files from this repository, record the version used, and preserve each processed image’s mouse identifier, slice index, and timeframe index so our preparation procedure remains traceable and reproducible. 

## License

The Open-Source Pre-Clinical Segmentation GitHub repository is distributed under the GNU General Public License version 3.0 (GPL-3.0) [2]. This license permits use, modification, and redistribution of covered materials subject to its terms, including applicable attribution and source-distribution requirements. Our project will use the repository for academic research, acknowledge the original authors, and comply with the applicable licensing conditions. We will confirm whether any separate licensing restrictions apply to the MRI dataset before redistributing its files.

## Ethical Statements 

Our project uses existing mouse cardiac MRI data and does not involve new animal experiments or human participant information. We will acknowledge the original researchers’ data collection and animal research oversight. We will report exclusions, preprocessing decisions, and model limitations transparently, and we will not present automated segmentations as substitutes for expert assessment without additional validation. 
