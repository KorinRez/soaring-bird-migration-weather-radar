# Third-Party Code Attribution

This directory contains code adapted from external sources used in my radar data processing pipeline.

---

## Source Repository

**Original Author**: Ilya Savenko  
**Repository**: [radar-bird-segmentation](https://github.com/ilyasa332/radar-bird-segmentation)  
**Access Date**: [January 15, 2025]  
**License**: [CHECK REPOSITORY - MIT/BSD/GPL/None]  
**Status**: Unpublished research code

---

## Components Used

### 1. PPI Image Generation Code (Section 1.2)

**Location in our pipeline**: `1_hdf_to_ppi.py`, Section 1.2  
**Purpose**: Convert HDF5 radar data files to PPI (Plan Position Indicator) image format

**Functions/Classes**:
- `RadarScan()`: Main class for converting HDF radar data to PPI images
  - Parameters: file path, elevation angle, site name, data type, pixel resolution
  - Output: Saves PPI images in specified format

**Original Source**: Lines adapted from Ilya Savenko's processing pipeline  
**Modifications**: 
- Adapted for Israeli Meteorological Service (IMS) radar data format
- Modified to process multiple elevation angles (0.5° - 2.2°)
- Added error handling for corrupted files
- Integrated with our daytime filtering and file size checks

---

### 2. mini_utils.py

**Purpose**: Utility functions for radar file handling and data organization

**Functions Used**:
- `getFileTime(file)`: Extracts timestamp from radar filename
- `groupTimewise(files)`: Groups radar files by date for batch processing
- `getSaveDir(file, folder, elevation)`: Generates output directory paths
- `getFolderDir(file)`: Creates folder structure based on file date
- Additional helper functions for HDF file manipulation

**Modifications**: 
- [If NO changes:] Used as-is from original repository
- [If changes made:] 
  - Modified `[function name]` to [description]
  - Added [feature] to `[function name]`

---

## Version Information

**Snapshot Date**: [DATE YOU COPIED THE CODE]

**Important Note**: This is a snapshot of Ilya Savenko's code from [DATE]. The original repository may have been updated since this version was copied. For the most current version, please visit:  
https://github.com/ilyasa332/radar-bird-segmentation

**Git Commit** (if known): [INSERT COMMIT HASH, e.g., abc123def456]

---

## How We Use This Code

1. **Data Preprocessing** (`1_hdf_to_ppi.py`):
   - Read raw HDF5 radar files from IMS
   - Filter for daytime periods and valid file sizes
   - Convert to PPI images for each elevation angle
   - Organize output by date and elevation

2. **Batch Processing**:
   - Group files by date using `groupTimewise()`
   - Process each batch sequentially
   - Save results in organized directory structure

3. **Integration with Our Pipeline**:
   - PPI images → CNN-based flock detection (Schekler et al. 2023)
   - Detection results → Bird density quantification
   - Final output → Migration characterization

---

## Citation

If you use this code or our adaptation of it, please cite:

**Original Code**:
