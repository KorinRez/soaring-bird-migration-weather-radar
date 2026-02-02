# soaring-bird-migration-weather-radar

This repository contains the workflow for quantifying and characterizing diurnal soaring bird migration, developed by Reznikov et al. at the University of Haifa, Israel.

For details, see our publication:

[Citation will be added upon publication]

---

## Workflow Overview

This pipeline processes weather radar data to detect and quantify soaring bird migration through four main steps:

### 1. Radar fi Preprocessing (`1_hdf_to_ppi.py`)


1. 1_hdf_to_ppi
2. 2_prediction:
Download our best epoch from: (https://campushaifaac-my.sharepoint.com/:u:/g/personal/krezni01_campus_haifa_ac_il/IQBppZnhDiVVRKuDsU_pgMxOAQLqM4hXFks6qBV7GQc7kFY?e=hQLhJu)


## Acknowledgments

This code uses components from the following repositories:

- **1_HDF_TO_PPI: Creating PPI images**: Code adapted from [Ilya Savenko's radar-bird-segmentation](https://github.com/ilyasa332/radar-bird-segmentation)
- **2_prediction: Using UNET flocks detection model**: Code adapted from [Inbal Sheckler'sUNET-flocks-detection](https://github.com/Inbal-Schekler/UNET-flocks-detection/tree/main)

