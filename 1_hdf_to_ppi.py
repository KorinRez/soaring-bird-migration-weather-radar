 # --------------------------------------- HDF TO PPI RadarScan ---------------------------------------

# ----------------------------------------------------------------------------------------------------
#                                     
#                                       SETTING ENVIRONMENT
#
# ----------------------------------------------------------------------------------------------------
import sys
import pickle

# For sunrise/set times
from astral import LocationInfo
from astral.sun import sun
from datetime import datetime, time

from Flight_properties_functions import is_daytime_file

# TODO: modify as needed - this is the path to mini_utils by Ilya Savenko's radar-bird-segmentation repository (Original source: https://github.com/ilyasa332/radar-bird-segmentation)
sys.path.append(r'my\path\to\mini_utils')
from mini_utils import *

from tqdm import tqdm


def main():
    try:
        print("Running script 1_hdf_to_ppi.py")

# ----------------------------------------------------------------------------------------------------
#
#                                        DEFINE PARAMETERS
#  
# ----------------------------------------------------------------------------------------------------
        # TODO: Adjust the parameters below according to your needs
      
        SITE = 'name_of_your_site'  
        ELEVATIONS = [1, 2, 3, 4, 5, 6]  
        year = '2025' 
        months = ['08', '09', '10', '11']  
        months_str = ['august', 'september','october', 'november']  # Change according to your needs

        for m, m_str in zip(months, months_str):
            print(f"\n=== Processing {m_str.upper()} ===")
            month = m
            year_month = f'{year}_{m_str}'
            print(month, year_month)

            DATA_TYPE = 'VRAD'  # Input VRAD, DBZ, HDBZ, WRAD for desired parameter
            PIXEL_RESOLUTION = 256

            lat =  00.000 # Add the radar latitude
            long =  00.000 # Add the radar longtitude
            timezone = 'Asia/Jerusalem'  # Add your timezone

            location = LocationInfo(SITE, timezone, timezone, lat, long)

            # Minimum file size in bytes (4000 KB = 4,000,000 bytes) 
            min_file_size = 4000 * 1024 # TODO: Modify as needed 

            # TODO: Modify as needed - Get the HDF folder path.  
            hdf_folder = fr'My:/PATH_TO_RADAR_FILES/H5/{year}/{month}/'  # files names should be in the following format:0000ISR-PPIVol-20240909-000003-0a0a.hdf

            # TODO: Modify as needed - Set saving settings for where the PPI images will be saved
            save_dir = fr'MY:\PATH\TO_PPI_IMAGES\HDF_to_PPI\ppi_vrad\{SITE}\{year}\{month}'
            save_phase = ''  # Can be left empty

# ----------------------------------------------------------------------------------------------------
#
#                                          HOUSEKEEPING
#          
# ----------------------------------------------------------------------------------------------------
         # Filter files (daytime & size)
            hdf_paths = []
            for root, _, files in os.walk(hdf_folder):
                for file in files:
                    if file.endswith('.hdf'):
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        if file_size > min_file_size and is_daytime_file(file, location):
                            hdf_paths.append(file_path)
                         
            # Sort files
            sorted_files = sorted(hdf_paths, key=lambda file: getFileTime(file))  

            # Export sorted_files - needed for script 3_extract_ppi_data
            file_name = f"{SITE}_{year_month}_h5.pkl"

            with open(fr'MY:\PATH\TO\daytime_h5_list\{file_name}', 'wb') as f:  
                pickle.dump(sorted_files, f)

            # Split the files to different dates
            grouped_files = groupTimewise(sorted_files)
            # Print lengths of the lists
            print(f"Number of files: {len(hdf_paths)} \n" +
                f"Number of groups: {len(grouped_files)}")

# ---------------------------------------------------------------------------------------------------
#          
#                               CREATING PPI IMAGES
#      
#          Code adapted from: Ilya Savenko's radar-bird-segmentation repository
#          Original source: https://github.com/ilyasa332/radar-bird-segmentation  
# ----------------------------------------------------------------------------------------------------
"""
Copyright (c) 2025 Ilya Savenko
Licensed under the MIT License

Original Repository: https://github.com/ilyasa332/radar-bird-segmentation
See LICENSE file in this directory for full license text.
"""
            # Create images for all batches of HDFs
            for i, batch in enumerate(grouped_files):
                # Folder for each batch
                save_folder = os.path.join(save_dir, save_phase, getFolderDir(batch[0]))
                # Save for each HDF the images - for elevations from the lowest one to ELEVATIONS
                print(f"Saving batch number {i + 1} out of {len(grouped_files)} to {save_folder}")
                for file in tqdm(batch):
                    for elev in ELEVATIONS:
                        # Get save directory
                        saveDir = getSaveDir(file, save_folder, elev)
                        # Create directory
                        os.makedirs(os.path.dirname(saveDir), exist_ok=True)
                        try:
                            RadarScan(file, elev, site=SITE, data_type=DATA_TYPE, pixelResolution=PIXEL_RESOLUTION).save(
                                saveDir)
                        except Exception as e:
                            print(f"Error occurred while processing {file} at elevation {elev}: {e}")
                            continue

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)  # Exit with a non-zero status code to indicate an error

if __name__ == "__main__":
    main()

