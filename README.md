# soaring-bird-migration-weather-radar

This repository contains the workflow for quantifying and characterizing diurnal soaring bird migration from weather radar files, developed by Reznikov et al. at the University of Haifa, Israel.

For details, see our publication:

[Citation will be added upon publication]

---

## Workflow Overview

### 1. PPI Images Creation (`1_hdf_to_ppi.py`)

**Purpose**: Create PPI (Plan Position Indicator) images from raw HDF5 radar files  

**What it does**:
- Filters radar files for daytime periods and valid file sizes
- Extracts radar scans at specified elevation angles 
- Create PPI images suitable for the UNET-flocks-detection model, processing from HDF5 format files
- Organizes output by date and elevation angle

**Input**: Raw HDF5 radar files

**Output**: PPI images organized in directories by date and elevation

**Code Attribution**: 
This script uses radar preprocessing utilities from [Ilya Savenko's radar-bird-segmentation](https://github.com/ilyasa332/radar-bird-segmentation) (MIT License). See [`third_party/mini_utils/README.md`](third_party/mini_utils/README.md) for details.

---

### 2. Flocks Detection (`2_prediction.py`)

**Purpose**: Detect migrating soaring bird flocks in VRAD PPI images using deep learning U-Net model developed by Schekler et al. (2023)

**What it does**: Applies trained CNN U-Net model, developed by Schekler et al. (2023), to identify bird flocks segments

**Prerequisites**:
1. **Download the trained model weights**:
   - [Download best_epoch model](https://campushaifaac-my.sharepoint.com/:u:/g/personal/krezni01_campus_haifa_ac_il/IQBppZnhDiVVRKuDsU_pgMxOAQLqM4hXFks6qBV7GQc7kFY?e=hQLhJu)
  
**Input**: VRAD PPI images from Step 1  
**Output**: Detection arrays with dates

**Code Attribution**: 
This script uses the flock detection model from [Inbal Schekler's UNET-flocks-detection](https://github.com/Inbal-Schekler/UNET-flocks-detection). Based on Schekler et al. (2023) *Methods in Ecology and Evolution*, 14, 2084-2094. See See [`third_party/UNET-flocks-detections-functions/README.md`](third_party/UNET-flocks-detections-functions/README.md) for details..



