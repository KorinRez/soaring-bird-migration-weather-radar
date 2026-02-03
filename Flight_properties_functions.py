import numpy as np
import os
import gc
import re
from tqdm import tqdm  

from astral.sun import sun
from datetime import datetime, time

from pyproj import CRS, Transformer  

# Functions to process PPI in Python using bioRad package in R:
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

# Import R's bioRad package
bioRad = importr("bioRad")

#------- use in 1_hdf_to_ppi.py ------
def is_daytime_file(file_name, radar_location):
    # File name should have date time: 0000ISR-PPIVol-20140427-144102-0a0a
    match = re.search(r'(\d{8})-(\d{6})', file_name)
    if match:
        time_str = match.group(2)
        print(time_str)
        hour, minute, second = int(time_str[:2]), int(time_str[2:4]), int(time_str[4:6])
        file_time = time(hour, minute, second)

        # Extract date from the file name
        date_str = match.group(1)
        print(date_str)
        file_date = datetime.strptime(date_str, "%Y%m%d").date()

        # Calculate sunrise and sunset for that date
        sun_times = sun(radar_location.observer, date=file_date)
        sunrise = sun_times['sunrise'].time()
        sunset = sun_times['sunset'].time()

        # Check if the file time is within daytime
        return sunrise <= file_time <= sunset
    return False

#-------  use in 3_extracting_ppi_metadata.py ------


def sort_by_date_time(data_list):
    """
    Sort a list of dictionaries by the 'date' and 'time' keys.

    Parameters:
    - data_list: List of dictionaries with 'date' and 'time' keys.

    Returns:
    - A sorted list of dictionaries.
    """
    try:
        # Use sorted with a custom key combining 'date' and 'time'
        sorted_list = sorted(
            data_list,
            key=lambda x: (x['date'], x['time'])  # Sort by date first, then time
        )
        return sorted_list
    except KeyError as e:
        print(f"Missing key during sorting: {e}")
        return data_list


def loadRelevantFiles(prediction_dict, h5_files):
    relevant_files = []
    for item in prediction_dict:
        date_bird_pred = item['date']
        time_bird_pred = item['time']

        for file in h5_files:
            date_h5 = os.path.basename(file).split('-')[2]
            # print(date_h5)
            time_h5 = os.path.basename(file).split('-')[3]
            # print(time_h5)

            if date_bird_pred == date_h5 and time_bird_pred == time_h5:
                relevant_files.append(file)
                break
    return relevant_files


def extract_proj4_from_first_file(file_path, grid_size=250, range_max=50000):
    read_pvolfile = ro.r("read_pvolfile")
    project_as_ppi = ro.r("project_as_ppi")

    pvol = read_pvolfile(file_path, elev_min=-1, param="DBZH")
    scan = pvol.rx2("scans")[0]
    ppi = project_as_ppi(scan, grid_size=grid_size, range_max=range_max)
    data = ppi.rx2("data")

    # Extract the projection string
    proj4string_obj = data.slots["proj4string"]
    proj4_string = ro.r('slot')(proj4string_obj, "projargs")
    # print(f"Extracted Proj4 String: {proj4_string}")

    # Convert R StrVector to a normal Python string
    proj4_string = str(proj4_string[0])
    print(f"Converted Proj4 String: {proj4_string}")

    return proj4_string


def prepare_projection_and_grid(proj4_string, range_max, array_size):
    """
    Precompute projection and coordinate grid to avoid repeating it for each scan.
    """
    crs_aeqd = CRS.from_string(proj4_string)
    crs_wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_aeqd, crs_wgs84, always_xy=True)

    x_grid = np.linspace(-range_max, range_max, array_size)
    y_grid = np.linspace(-range_max, range_max, array_size)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
    lon_grid, lat_grid = transformer.transform(X, Y)

    # Flip lat/lon so that row 0 is north
    lat_grid = np.flipud(lat_grid)
    lon_grid = np.flipud(lon_grid)

    return lon_grid, lat_grid, transformer


def process_PPI_file(file_path, scan_num, grid_size, range_max, array_size, lon_grid, lat_grid):
    """
    Extract dBZ and lat/lon grid for a single radar scan file as a PPI data.
    """
    try:
        # Import R functions
        read_pvolfile = ro.r("read_pvolfile")
        project_as_ppi = ro.r("project_as_ppi")

        # Load radar file and extract scan
        pvol = read_pvolfile(file_path, elev_min=-1, param="DBZH")
        scan = pvol.rx2("scans")[scan_num]
        ppi = project_as_ppi(scan, grid_size=grid_size, range_max=range_max)
        data = ppi.rx2("data")

        # Get raw reflectivity data
        dbz_raw = np.array(data.slots["data"])
        dbz = dbz_raw[0] if dbz_raw.ndim == 2 and dbz_raw.shape[0] > 1 else dbz_raw
        dbz = np.nan_to_num(dbz.flatten(), nan=256)

        # Reshape logic
        expected_size = array_size ** 2
        if dbz.shape != (expected_size,):
            dbz = dbz[:-array_size]
        dbz = dbz.reshape((array_size, array_size))

        # Get grid metadata (used to calculate distance from radar)
        grid = data.slots["grid"]
        x_coords = np.array(
            grid.slots["cellcentre.offset"][0] + grid.slots["cellsize"][0] * np.arange(grid.slots["cells.dim"][0]))
        y_coords = np.array(
            grid.slots["cellcentre.offset"][1] + grid.slots["cellsize"][1] * np.arange(grid.slots["cells.dim"][1]))

        return {
            'date': os.path.basename(file_path).split('-')[2],
            'time': os.path.basename(file_path).split('-')[3].split('_')[0],
            'dBZ': dbz,
            'x_coords': x_coords,
            'y_coords': y_coords,
            'lon_grid': lon_grid,
            'lat_grid': lat_grid
        }

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None


def process_PPI_files_in_chunks(file_paths, scan_num, proj4_string, grid_size=250, range_max=50000,
                                array_size=400, chunk_size=200):
    """
    Process radar files in memory-safe chunks to avoid memory crashes.
    """
    lon_grid, lat_grid, _ = prepare_projection_and_grid(proj4_string, range_max, array_size)

    all_results = []

    num_chunks = (len(file_paths) + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, len(file_paths))
        file_chunk = file_paths[chunk_start:chunk_end]

        print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} with {len(file_chunk)} files...")

        chunk_results = []
        for path in tqdm(file_chunk, desc=f"Chunk {chunk_idx + 1}"):
            result = process_PPI_file(
                path, scan_num, grid_size, range_max, array_size, lon_grid, lat_grid
            )
            if result is not None:
                chunk_results.append(result)

            gc.collect()  # Free RAM after each file

        all_results.extend(chunk_results)

        # Cleanup after chunk
        del chunk_results
        gc.collect()

    return all_results
