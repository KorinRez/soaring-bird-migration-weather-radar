"""
Copyright (c) 2025 Inbal Schekler
Licensed under the MIT License

Original Repository: https://github.com/Inbal-Schekler/UNET-flocks-detection
See LICENSE file in this directory for full license text.
"""

import numpy as np
from PIL import Image
import os
from create_previous_images import create_early_image_2


def image_generator(files, num_past, minuts, batch_size=32, sz=(256, 256)):
    while True:
        # extract a random batch
        batch = np.random.choice(files, size=batch_size)

        # variables for collecting batches of inputs and outputs
        batch_x = []
        img_time = []

        for f in batch:

            # preprocess the raw images
            raw = Image.open(f)
            raw = raw.resize(sz)
            raw = np.array(raw)

            prev_image = create_early_image_2(files, f, num_past, minuts, sz)
            if len(prev_image) == 0:

                continue
            else:
                raw_prev = np.concatenate((raw, prev_image), axis=2)

                # Create a dictionary to relate date and time to each img
                img_time_dict = {'date': str(os.path.basename(f).split('-')[2]),
                                 'time': str(os.path.basename(f).split('-')[3].split('_')[0]),
                                 'img': raw/255}

            img_time.append(img_time_dict)
            batch_x.append(raw_prev)

        # preprocess a batch of images and masks
        batch_x = np.array(batch_x) / 255.

        yield (batch_x, img_time)






