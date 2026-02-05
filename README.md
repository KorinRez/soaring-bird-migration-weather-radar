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

**Output**: 
- list of filterd radar files (.pkl files)
- PPI images organized in directories by date and elevation (.Tiff files)

**Code Attribution**: 
This script uses functions and code from [Ilya Savenko's radar-bird-segmentation](https://github.com/ilyasa332/radar-bird-segmentation) (MIT License). See [`third_party/mini_utils/README.md`](third_party/mini_utils/README.md) for details.

---

### 2. Flocks Detection (`2_prediction.py`)

**Purpose**: Detect migrating soaring bird flocks in Radial velocity (VRAD) PPI images 

**What it does**: Applies trained CNN U-Net model, developed by Schekler et al. (2023), to distinguish patterns of migrating soaring bird flocks from other targets detected by the radars, such as wide-­front passerine migration, ground clutter, and rain clouds.

**Prerequisites**:
1. **Download the trained model weights**: [Download best_epoch model](https://campushaifaac-my.sharepoint.com/:u:/g/personal/krezni01_campus_haifa_ac_il/IQBppZnhDiVVRKuDsU_pgMxOAQLqM4hXFks6qBV7GQc7kFY?e=hQLhJu)
  
**Input**: VRAD PPI images (256×256 pixels) from Step 1 (.Tiff files)

**Output**: Flock probability arrays (256×256, values 0-1 per pixel) with timestamps (.pkl files)

**Code Attribution**: 
This script uses the flock detection model from [Inbal Schekler's UNET-flocks-detection](https://github.com/Inbal-Schekler/UNET-flocks-detection). Based on Schekler et al. (2023) *Methods in Ecology and Evolution*, 14, 2084-2094. See See [`third_party/UNET-flocks-detections-functions/README.md`](third_party/UNET-flocks-detections-functions/README.md) for details..

---

### 3. Extract Radar Parameters (`3_extracting_ppi_metadata.py`)

**Purpose**: Extract radar parameters (reflectivity, coordinates), compute distance grid data and integrate with flock detection results

**What it does**: 
- Extracts dBZ (reflectivity factor) and geographic coordinates from original radar files using the functions read_pvolfile and project_as_ppi from bioRad package.
- Removes duplicate detections
- Resizes flock probability arrays from 256×256 to 400×400 to match radar data spatial resolution
- Converts probability values (0-1) to binary predictions (flock/non-flock) using 0.5 threshold
- Calculates Euclidean distance from radar to each grid cell
- Integrates all parameters (detection, dBZ, coordinates, distance) by timestamp.

**Prerequisites**:
1. **R and bioRad package**: Install R and the bioRad package
2. **Python rpy2 interface**: `pip install rpy2` to call R functions from Python
  
**Input**: 
- list of filterd radar files from Step 1 (.pkl files)
- Flock probability arrays from Step 2 (.pkl files)

**Output**: 
Integrated datasets (.joblib files) containing:

- Binary flock predictions (400×400 grid)
- dBZ values (400×400 grid)
- Geographic coordinates (lat/lon ; 400×400 grid)
- Distance from radar (400×400 grid)
- Timestamps

**Output**: Implements `read_pvolfile` and `project_as_ppi` functions from bioRad R package (Dokter et al. 2019) via Python rpy2 interface.



