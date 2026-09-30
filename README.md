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
- List of filterd radar files (.pkl files)
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
- List of filterd radar files from Step 1 (.pkl files)
- Flock probability arrays from Step 2 (.pkl files)

**Output**: Integrated datasets (.joblib files) containing:
- Binary flock predictions (400×400 grid)
- dBZ values (400×400 grid)
- Geographic coordinates (lat/lon ; 400×400 grid)
- Distance from radar (400×400 grid)
- Timestamps

***Code Attribution**: Implements `read_pvolfile` and `project_as_ppi` functions from bioRad R package (Dokter et al. 2019) via Python rpy2 interface.

---

### 4. Filtering, Quantification and Height Analysis (`4_filtering_quantification_analyses.py`)

**Purpose**: Filter out non-biological targets (clouds, residual clutter), convert detections into bird counts, and aggregate individual detections into flock-level clusters with migration height and region information.

**What it does**:
- Builds a land/sea mask for the radar grid (from a country boundary GeoJSON) to exclude detections over the sea, with a regional correction step to fix inland areas misclassified as sea
- Applies combined per-pixel filtering: dBZ range, model prediction, distance from radar (≤50 km), and the sea mask
- Converts filtered dBZ values to reflectivity
- Removes residual non-biological targets (mainly clouds) using empirically tuned, per-cluster thresholds (cluster size, width, reflectivity sum and ratio), with separate thresholds for September (peak migration month) vs. other months
- Computes the height range of each detection 
- Converts reflectivity to bird counts using a date-matched RCS (Radar Cross-Section) T-matrix value
- Spatially clusters detections within each scan to identify individual flocks, and computes a Gaussian-weighted height distribution per cluster
- After all elevations for a given month are processed: combines all elevation files, removes duplicate detections across overlapping elevation angles (keeping the highest bird count per voxel), and aggregates pixels into cluster-level records (summed reflectivity/bird count, mean distance, centroid coordinates)
- Converts height from above see level (ASL) to above ground level (AGL) using a digital elevation model raster, and keeps only clusters whose flock top is at least 200 m AGL
- Assigns each cluster to a north/south region relative to the radar site
- Exports a final, cluster-level CSV per site and month

**Prerequisites**:
1. **Conda environment** *(optional — this is the environment used for this analysis; not a strict requirement)*:
 conda create -n bird_radar_env -c conda-forge python=3.9 rasterio shapely pandas numpy scipy joblib matplotlib h5py astral pyproj tqdm rpy2 cartopy

2. **Reference files** (paths set at the top of the script):
   - Country-boundary GeoJSON, for sea masking
   - `mean_RCSs.csv` — mean RCS (T-matrix) values by date range, used to convert reflectivity to bird counts
   - DEM raster (`.tif`), for computing height AGL
  
   **Notes**:
  - The sea mask (and its regional correction) is specific to this study's geographic area and radar site; it is optional and should be adjusted, replaced, or    removed for other regions.
  - The per-cluster cloud-filtering step (removing residual non-biological targets by empirically tuned thresholds) is optional and can be removed or adjusted if     false-positive detections from the Flock Detection Model are filtered out in another way.
  - The RCS values used for the bird-count conversion can be calculated following the method described in Reznikov et al. (2025), *J. R. Soc. Interface*, 22(231), 20250510. https://doi.org/10.1098/rsif.2025.0510

**Input**: PPI metadata (.joblib files) and radar file metadata (.json files) from Step 3, per elevation angle

**Output**: 
- Final cluster-level (each cluster represent a flock) CSV per site/month, containing: date, time, cluster location (centroid coordinates), height range (ASL and AGL), bird count, reflectivity, cluster size and width, distance from radar, and region (north/south)
