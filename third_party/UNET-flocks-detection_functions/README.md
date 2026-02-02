# Third-Party Code Attribution

This directory contains code from external sources used in my radar data processing pipeline.

---

## Source Repository

**Original Author**: Inbal Schekler  
**Repository**: [UNET-flocks-detection](https://github.com/Inbal-Schekler/UNET-flocks-detection)  
**Access Date**: [January 15, 2023]  
**License**: MIT License (see `LICENSE.txt`)  
**Status**: Open-source research code

---

## Components Used

Utility functions for prediction preprocessing:
- `unet()`: UNET model for predicting soaring flocks
- `image_generator()`, `(create_early_image_2)`: images handling utilities for the UNET model

All code is used unmodified from the original repository.

---

## citation
Schekler I, Nave T, Shimshoni I, Sapir N. 2023 Automatic detection of migrating soaring bird flocks using weather radars by deep learning. Methods Ecol. Evol. 14, 2084–2094. (doi:10.1111/2041-210x.14161)

---

## Acknowledgment

We thank Inbal Schekler for making her UNET Flocks Detection Model publicly available under the MIT License, which greatly facilitated the development of my migration monitoring pipeline.
