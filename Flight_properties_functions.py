import numpy as np
import os
import gc
import re
import math as math
import h5py

from tqdm import tqdm 
from astral.sun import sun
from datetime import datetime, time
from scipy.stats import norm
from pyproj import CRS, Transformer  

# Functions to process PPI in Python using the bioRad package in R:
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


#-------  use in 3_extracting_ppi_metadata.py & 4_filtering_quantification_analyses.py------

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

#-------  use in 3_extracting_ppi_metadata.py ------

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
    Precompute the projection and coordinate grid to avoid repeating it for each scan.
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
    Extract dBZ and lat/lon grid for a single radar scan file as PPI data.
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


def Cartesian_distance_to_Euclidean(dictionary):
    """
    meters units
    Cut y_coords to match the size of x_coords and calculate the Euclidean distance array.
    :param dictionary: contain 'x_coords' and 'y_coords' that were calculated in def process_PPI_files_in_chunks
    :return: the dict with a new key called 'euclidean_distance'
    """
    x_coords = dictionary['x_coords']
    y_coords_original = dictionary['y_coords']

    # Adjust y_coords to ensure it matches x_coords (I cut the last row of the dbz as well, so it's fine - this is an extra not needed row)
    y_coords = y_coords_original[:400]

    # Create 2D grids for x and y
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # Calculate the Euclidean distance from the radar
    distance_array = np.sqrt(x_grid ** 2 + y_grid ** 2)

    # Add the new key to the dictionary
    dictionary['euclidean_distance'] = distance_array
    return dictionary


def merge_dictionaries(first_list, second_list):
    """
    Merge three lists of dictionaries based on 'date' and 'time' keys.

    Parameters:
    - first_list: List of dictionaries with 'date', 'time', and 'dBZ' data.
    - second_list: List of dictionaries with 'date', 'time', and prediction data.

    Returns:
    - merged_list: List of merged dictionaries.
    """
    # Create a dictionary with date and time as the name for each list for faster lookup
    dbz_dict = {(d['date'], d['time']): d for d in first_list}
    bird_pred_dict = {(d['date'], d['time']): d for d in second_list}

    merged_list = []

    # Iterate over dbz_list and merge with corresponding dictionaries
    for (date, time), dbz_entry in dbz_dict.items():
        merged_entry = dbz_entry.copy()  # Start with dbz dictionary

        # Add bird_prediction data if available
        if (date, time) in bird_pred_dict:
            merged_entry.update(bird_pred_dict[(date, time)])

        merged_list.append(merged_entry)

    return merged_list   
    
#-------  use in 4_filtering_quantification_analyses.py ------

def dBZ_to_reflectivity(dictionary, radar_wavelength=5.3, refracIndex_K=0.93):
    # Define ref_constant for the conversion
    numerator = (10 ** 3) * (math.pi ** 5)
    denominator = radar_wavelength ** 4
    ref_constant = (numerator / denominator) * refracIndex_K

    dbzValues_filter = dictionary['filtered_prediction_dBZ_0']

    dbzValues_masked_filter = np.ma.masked_where(dbzValues_filter == 256, dbzValues_filter)

    # Convert from dB scale (dBZ units) to reflectivity factor (small z)
    reflectivity_factor_filter = 10 ** (dbzValues_masked_filter / 10)

    # convert from reflectivity_factor in units of mm^6/m^3 to reflectivity_eta in units of cm^2/km^3:
    # ref_constant is the proportionality constant between reflectivity_factor z in mm ^ 6 / m ^ 3 and
    # reflectivity_eta in cm ^ 2 / km ^ 3, when radar Wavelength in cm:
    reflectivity_eta_filter = reflectivity_factor_filter * ref_constant

    dictionary['prediction_reflectivity_eta_0'] = reflectivity_eta_filter.filled(0)  # all masked values filled with 0

    # sum the unmasked values
    dictionary['reflectivity_sum_0'] = reflectivity_eta_filter.sum()

def compute_max_horizontal_width(row_indices, col_indices):
    rows_to_cols = defaultdict(list)
    for r, c in zip(row_indices, col_indices):
        rows_to_cols[r].append(c)
    return max((max(cols) - min(cols) + 1 if len(cols) > 1 else 1) for cols in rows_to_cols.values())

def height_range(h5_file, dictionary, radar_height, elev_predict, angle_resolution=1, earth_radius=(4 / 3) * 6374):
    """
    The function work on one dictionary. if it is a list of dictionaries, it should be inside a loop.
    calculate max height above sea level. Based on the formula h= sqrt(d^2 + Re^2 +2dResin(θ))−Re. Re-radius of earth
    extract file anfle and calculate the max height in each cell.
    :param h5_file: list of hf files
    :param dictionary: the dictionary with the relevant key of 'euclidean_distance' im meters
    :param radar_height: the height of the radar above sea level in km
    :param elev_predict: the elevation angle index to extract from the h5 file
    :param angle_resolution: the azimuth range of the radar
    :param earth_radius: the earth radius
    :return: Modifies the dictionary in place by adding:
             - 'angle': The scan angle.
             - 'max_height': Array with maximum height at each distance.
             - 'min_height': Array with minimum height at each distance.
             - 'height_range': A categorical array with height ranges in string format.
    """
    try:
        with h5py.File(h5_file, 'r') as f:
            # Access the dataset and the 'where' group
            dataset_name = f[f'dataset{elev_predict}']
            where_group = dataset_name['where']
            angle = where_group.attrs['elangle']  # Extract elevation angle
    except Exception as e:
        print(f"Error reading H5 file '{h5_file}': {e}")
        return None

    dictionary['angle'] = float(angle)  # Store angle in the dictionary

    try:
        d_km = dictionary['euclidean_distance'] / 1000  # Convert meters to kilometers
        Re = earth_radius

        # Convert elevation angle to radians
        angle_rad = np.radians(angle)
        half_rad = np.radians(angle_resolution / 2)

        sin_max = np.sin(angle_rad + half_rad)
        sin_min = np.sin(angle_rad - half_rad)

        # Compute max height (upper beam limit)
        h_max = radar_height + np.sqrt(
            d_km ** 2 + Re ** 2 + 2 * d_km * Re * sin_max) - Re + 0.015  # Adds a correction 0.015 (~15m) for beam spreading

        # Compute min height (lower beam limit)
        h_min = radar_height + np.sqrt(d_km ** 2 + Re ** 2 + 2 * d_km * Re * sin_min) - Re + 0.015

        dictionary['max_height'] = h_max.astype(np.float32, copy=False)  # Max height in km
        dictionary['min_height'] = h_min.astype(np.float32, copy=False) # Min height in km

    except Exception as e:
        print(
            f"Error computing height range for dictionary with time '{dictionary.get('time')}' and date '{dictionary.get('date')}': {e}")
        return None

    return None


def run_height_range_in_chunks(h5_files, scan_dicts, radar_height, elev_predict, chunk_size=50):
    """
    Processes height range in memory-efficient chunks
    """
    results = []
    total = len(h5_files)
    num_chunks = (total + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, total)
        h5_chunk = h5_files[chunk_start:chunk_end]
        dict_chunk = scan_dicts[chunk_start:chunk_end]

        print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} ({len(h5_chunk)} files)...")

        for f, d in tqdm(zip(h5_chunk, dict_chunk),
                         desc=f"Chunk {chunk_idx + 1}", total=len(h5_chunk), leave=False):
            height_range(f, d, radar_height, elev_predict)
            results.append(d)
            gc.collect()  # force cleanup of temps inside height_range

        gc.collect()

    return results
