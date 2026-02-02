"""
Copyright (c) 2025 Inbal Schekler
Licensed under the MIT License

Original Repository: https://github.com/Inbal-Schekler/UNET-flocks-detection
See LICENSE file in this directory for full license text.
"""


from datetime import datetime
import re
import os
import numpy as np
from PIL import Image




def create_early_image_2(files, file, num_past, minuts,sz):
    """
    :param files: list of order images
    :param file: we check for previous images for this specific file
    :param num_past: number of previuos images to look for
    :param minuts: we will add the previous images only if the gap of time between each two images
                   is less than the minuts parametr

    :return: the image concatenate to the previous images we found
    """
    time = datetime.strptime(os.path.basename(file).split('-')[2] + ' ' + os.path.basename(file).split('-')[3], '%Y%m%d %H%M%S')
    index = files.index(file)
    time_prev = datetime.strptime(os.path.basename(files[index-1]).split('-')[2] + ' ' + os.path.basename(files[index-1]).split('-')[3], '%Y%m%d %H%M%S')
    file_prev = files[index - 1]
    delta = (time - time_prev).seconds / 60
    if delta < minuts:
        image_prev = Image.open(file_prev)
        image_prev = image_prev.resize(sz)
        image_prev = np.array(image_prev)

        if num_past > 1:
            time_prev_2 = datetime.strptime(os.path.basename(files[index-2]).split('-')[2] + ' ' + os.path.basename(files[index-2]).split('-')[3], '%Y%m%d %H%M%S')
            file_prev_2 = files[index - 2]
            delta_2 = (time_prev - time_prev_2).seconds / 60
            if delta_2 < minuts:
                image_prev_2 = Image.open(file_prev_2)
                image_prev_2 = image_prev_2.resize(sz)
                image_prev_2 = np.array(image_prev_2)

                image_prev_all = np.concatenate((image_prev, image_prev_2), axis=2)
            else:
                image_prev_all = []

    else:

        image_prev_all = []


    return image_prev_all
