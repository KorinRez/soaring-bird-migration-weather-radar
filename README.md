# soaring-bird-migration-weather-radar

This repository contains the workflow for quantifying and characterizing diurnal soaring bird migration, developed by Reznikov et al. at the University of Haifa, Israel.

For details, see our publication:

[Citation will be added upon publication]

---

## Workflow Overview

This pipeline processes weather radar data to detect, quantify and characterize soaring bird migration through four main steps:

### 1. (`1_hdf_to_ppi.py`)

**Purpose**: Create PPI (Plan Position Indicator) images from raw HDF5 radar files  

**What it does**:
- Filters radar files for daytime periods and valid file sizes
- Extracts radar scans at specified elevation angles 
- Create PPI images suitable for CNN processing from HDF5 format files
- Organizes output by date and elevation angle

**Input**: Raw HDF5 radar files
**Output**: PPI images organized in directories by date and elevation

**Usage**:
```bash
python 1_hdf_to_ppi.py
```


2. 2_prediction:
Download our best epoch from: (https://campushaifaac-my.sharepoint.com/:u:/g/personal/krezni01_campus_haifa_ac_il/IQBppZnhDiVVRKuDsU_pgMxOAQLqM4hXFks6qBV7GQc7kFY?e=hQLhJu)


## Acknowledgments

This code uses components from the following repositories:

- **1_HDF_TO_PPI: Creating PPI images**: Code adapted from [Ilya Savenko's radar-bird-segmentation](https://github.com/ilyasa332/radar-bird-segmentation)
- **2_prediction: Using UNET flocks detection model**: Code adapted from [Inbal Sheckler'sUNET-flocks-detection](https://github.com/Inbal-Schekler/UNET-flocks-detection/tree/main)

