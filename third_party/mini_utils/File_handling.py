"""
Copyright (c) 2025 Ilya Savenko
Licensed under the MIT License

Original Repository: https://github.com/ilyasa332/radar-bird-segmentation
See LICENSE file in this directory for full license text.
"""

import re
import os
import numpy as np
import h5py
from datetime import datetime, timedelta


def groupTimewise(files, COUNT_MAX = 40):
    files = files.copy()
    # Make sure files are sorted
    files = sorted(files, key=lambda path: getFileTime(path))
    # Group the files timewise
    batches = []
    temp_time = None
    curr = -1
    count = 1
    for file in files:
        # if temp_time is None or check_delta_times(get_time(file), temp_time) or count >= COUNT_MAX:
        if temp_time is None or checkDeltaTimes(getFileTime(file), temp_time):
            curr += 1
            count = 1
            batches.append([file])
        else:
            count += 1
            batches[curr].append(file)
        temp_time = getFileTime(file)
    return batches

def getFileTime(file: str, site='meron', **kwargs):
    site = kwargs.get('site', site)
    print(file)
    if site == 'eilat':
        time_str = re.search(r'(\d{8}\d{6})', os.path.basename(file))[0]
        if time_str is not None:
            return datetime.strptime(time_str, "%Y%m%d%H%M%S")
    if site == 'dagan':
        time_str = re.search(r'(\d{8}T\d{6})', os.path.basename(file))[0]
        if time_str is not None:
            return datetime.strptime(time_str, "%Y%m%dT%H%M%S")
    time_str = re.search(r'(\d{8}-\d{6})', os.path.basename(file))[0]
    if time_str is not None:
        return datetime.strptime(time_str, "%Y%m%d-%H%M%S")
    
def checkDeltaTimes(time1, time2, delta=20):
    # Calculate the time difference between the two datetime objects
    time_difference = abs(time1 - time2)

    # Define a 20-minute time threshold
    minutes = timedelta(minutes=delta)

    # Compare the time difference to the threshold
    return time_difference > minutes

def getFolderDir(file):
    fileTime = getFileTime(file)
    date = fileTime.strftime("%Y%m%d")
    time = fileTime.strftime("%H%M")
    folder = os.path.join(date, time)
    return folder

def getSaveDir(file, folder, elev, extension='.tiff'):
    name = getFileName(file)
    saveDir = os.path.join(folder, f'elev_{elev}', name)
    return saveDir + extension

def getFileName(file):
    name = os.path.basename(file)
    fileName, _ = os.path.splitext(name)
    return fileName

def folder_path_to_xlsx(folderPath):
    return folderPath.replace('\\', '_') + '.xlsx'

# Ilya code:
#def filterBirdMode(files):
#    filteredFiles = []
#    for file in files:
#        with h5py.File(file) as f:
#            if f['how'].attrs['task'] == b'BirdMode':
#                filteredFiles.append(file)
#    return filteredFiles

# My version to handles error
def filterBirdMode(files):
    filteredFiles = []
    for file in files:
        try:
            with h5py.File(file) as f:
                try:
                    if f['how'].attrs['task'] == b'BirdMode':
                        filteredFiles.append(file)
                        print('true')
                    else:
                        print(f" the file '{file}' is not in birdmode")
                except KeyError as e:
                    print(f"KeyError: Missing key in file '{file}': {e}")
                except AttributeError as e:
                    print(f"AttributeError: Missing attribute in file '{file}': {e}")
        except OSError as e:
            print(f"Error opening file '{file}': {e}")
    return filteredFiles


def filterWeatherMode(files):
    filteredFiles = []
    for file in files:
        try:
            with h5py.File(file) as f:
                try:
                    if f['how'].attrs['task'] == b'WeatherMode':
                        filteredFiles.append(file)
                        print('true')
                    else:
                        print(f" the file '{file}' is not in weathermode")
                except KeyError as e:
                    print(f"KeyError: Missing key in file '{file}': {e}")
                except AttributeError as e:
                    print(f"AttributeError: Missing attribute in file '{file}': {e}")
        except OSError as e:
            print(f"Error opening file '{file}': {e}")
    return filteredFiles

def filterBirdInverseMode(files):
    filteredFiles = []
    for file in files:
        try:
            with h5py.File(file) as f:
                try:
                    if f['how'].attrs['task'] == b'BirdMode_inverse':
                        filteredFiles.append(file)
                        print('true')
                    else:
                        print(f" the file '{file}' is not in BirdMode_inverse")
                except KeyError as e:
                    print(f"KeyError: Missing key in file '{file}': {e}")
                except AttributeError as e:
                    print(f"AttributeError: Missing attribute in file '{file}': {e}")
        except OSError as e:
            print(f"Error opening file '{file}': {e}")
    return filteredFiles


def get_time_mask(sorted_files, num_past=2, diff_minutes=7):
    # Get times array
    times = np.array([getFileTime(file) for file in sorted_files])
    # initialize mask
    time_mask = np.full(times.shape, True)
    # Time difference matrix
    times_diff = times[time_mask][:, None] - times[time_mask][None, :]
    # Find times where the time difference is bigger than diff_minutes and the time is after the other time
    time_mask = (times_diff < timedelta(minutes=diff_minutes)) & (times_diff > timedelta(minutes=0))
    # Take times where they had at least one fitting time behind it
    time_mask = np.sum(time_mask, axis=1) > 0
    # Take out times which their previous time didn't have a time before it which passed the times_diff filter
    for _ in range(num_past-1):
        time_mask = time_mask.astype(int)
        time_mask = time_mask - np.roll(time_mask, 1) == 0
    return time_mask


def find_hdf(hdf_folder, image_path):
    time = getFileTime(image_path)

#def filter
