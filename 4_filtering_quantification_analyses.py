# ----------------------------------------------------------------------------------------------------
#
#                                   Setting the environment
#
# ----------------------------------------------------------------------------------------------------

import copy
import gc
import glob
import json
import os
import time
from collections import defaultdict
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import label
from scipy.stats import norm
from shapely.geometry import MultiPoint, Point, shape
from shapely.prepared import prep

import Flight_properties_functions as fp

# ----------------------------------------------------------------------------------------------------
#
#                                        DEFINE PARAMETERS
#
## ----------------------------------------------------------------------------------------------------

# TODO: Adjust the parameters below according to your needs

# General
# =============
SITE = 'name_of_your_site'
ELEVATIONS = [1, 2, 3, 4, 5, 6]  
year = '2025'
months = ['08', '09', '10', '11']
months_str = ['august', 'september','october', 'november'] 

radar_height = 0.000

# RCS
# =============
mean_RCSs = pd.read_csv(r'MY\PATH\TO\mean_RCSs.csv')  # TODO: Modify path as needed 

# .Shape file for sea masking
# ===========================
# Load .GeoJSON file
with open(r"MY\PATH\TO\countries.geojson", encoding="utf-8") as f:  # TODO: Modify path as needed 
    geojson_data = json.load(f)

# Extract relevant site's geometry
country_geom_raw = None
for feature in geojson_data["features"]:
    if feature["properties"].get("ADMIN") == "your_countery_name" or feature["properties"].get("NAME") == "your_countery_name": # TODO: Modify "your_countery_name" as needed 
        country_geom_raw = shape(feature["geometry"])
        break

if country_geom_raw is None:
    raise ValueError("Countery geometry was not found in GeoJSON")

# Use prepared geometry for fast spatial operations
country_geom = prep(country_geom_raw)











# ////////////////////////////////////////////////////////////////////////////////////////////////////////

#                                           MAIN

# ////////////////////////////////////////////////////////////////////////////////////////////////////////

# Iterate the month
# ===============================================
for m, m_str in zip(months, months_str):
    month = m
    # month = "08"
    year_month = f'{year}_{m_str}'
    # year_month = "2024_august"
    print(month, year_month)

    all_cluster_meta = []

    # Iterate the elevation
    # ============================================
    for i, elev in enumerate(ELEVATIONS):
        elev_predict = ELEVATIONS[i]
        # elev_predict = ELEVATIONS[1]
        print(f'processing elevation {elev_predict}')

        # Uploading the PPI DATA
        print(f'Uploading the PPI DATA of {SITE} {year_month}')
        file_name = f"PPI_metadata_{SITE}_{year_month}.joblib"
        output_dir = fr'MY\PATH\TO\PPI_metadata/elev_{elev_predict}/'
        output_path = os.path.join(output_dir, file_name)

        if not os.path.exists(output_path):
            print(f"File not found: {output_path}. Skipping...")
            continue

        birds_data = joblib.load(output_path)

        gc.collect()

        birds_data = fp.sort_by_date_time(birds_data)

        """ Check dbz orientation; row 0 is north, so the orientation is ok
        array_test = birds_data[44]
        dbz_test = array_test['dBZ']

        # Create a base plot of the image
        from matplotlib import pyplot as plt

        plt.figure(figsize=(8, 8))
        plt.imshow(dbz_test, cmap="viridis")  # , origin="lower"; Display the image with the viridis colormap
        plt.savefig("unplipedDBZ.png")
        plt.close()
        
        # Check lat and long orientations are upside down cause it geographic array so row 0 is south and last row is north
        """

        # ----------------------
        #  Filters per pixel
        # ----------------------
        # Build land/sea mask for radar grid
        print("Building sea mask...")
        radar_latitudes = birds_data[0]['lat_grid']
        radar_longitudes = birds_data[0]['lon_grid']

        sea_mask = np.zeros_like(radar_latitudes, dtype=bool)

        for i in range(radar_latitudes.shape[0]):
            for j in range(radar_latitudes.shape[1]):
                point = Point(radar_longitudes[i, j], radar_latitudes[i, j])
                if not country_geom.contains(point):
                    sea_mask[i, j] = True  # True = sea

        # Make a copy to avoid modifying the original mask
        corrected_sea_mask = sea_mask.copy()

        # define Yehoda and Shomron as land
        for i in range(sea_mask.shape[0]):
            land_found = False
            for j in range(sea_mask.shape[1]):
                if not sea_mask[i, j]:  # Found land
                    land_found = True
                elif land_found:
                    # This pixel was sea, but is to the right of land
                    corrected_sea_mask[i, j] = False  # Reclassify as land

        """debug
        plt.imshow(corrected_sea_mask, cmap="gray")
        plt.title("Sea Mask")
        plt.colorbar(label="True = sea")
        plt.savefig('corrected_sea_mask.png')
        plt.close()
        """

        """debug
        for scan in birds_data:
            if "filtered_prediction_dBZ_0" in scan:
                del scan["filtered_prediction_dBZ_0"]
        """

        """debug
        for data in birds_data:
            dBZ = data.pop('dBZ')  # Original dBZ values
            prediction = data['pred_0_1']
            euclidean_dist = data['euclidean_distance']

            # Start with valid dBZ mask
            valid_dBZ_mask = (dBZ >= min_dBZ_threshold) & (dBZ < max_dBZ_threshold)

            # General filter mask: values that should be set to 256
            general_mask = (
                ~valid_dBZ_mask |
                (prediction == 0) |
                (euclidean_dist > max_distance)
            )

            # Sea mask filter: values that should be set to -10
            sea_mask = corrected_sea_mask & (prediction == 1)

            # Combine both filters into output
            filtered_dBZ = dBZ.copy()
            filtered_dBZ[general_mask] = 256
            filtered_dBZ[sea_mask] = -10  # Apply after general mask to override if needed

            # Save result
            data['filtered_prediction_dBZ_0'] = filtered_dBZ
            
        # Extract data for plotting
        target_day = '20140930'
        target_time = '085103'

        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        from matplotlib import colors
        import numpy as np

        target_scan = None
        for scan in birds_data:
            if scan.get('date') == target_day and scan.get('time') == target_time:
                target_scan = scan
                break
                
        prediction = target_scan["filtered_prediction_dBZ_0"]  # np.flipud()
        sea_mask_plot = corrected_sea_mask
        lon = target_scan["lon_grid"]
        lat = target_scan["lat_grid"]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([34.2, 35.4, 31.6, 32.5], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='10m')
        ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, alpha=0.5)

        # Overlay prediction data
        pred_cmap = plt.cm.viridis
        pred_cmap.set_bad(color="lightgray")
        masked_prediction = np.ma.masked_where(prediction < 0, prediction)
        ax.pcolormesh(lon, lat, masked_prediction, cmap=pred_cmap, shading="auto", alpha=0.8)

        # Overlay sea mask (make sea pixels semi-transparent blue)
        mask_overlay = np.where(sea_mask_plot, 1.0, np.nan)  # Sea = 1, land = nan
        ax.pcolormesh(lon, lat, mask_overlay, cmap='Blues', alpha=0.3, shading="auto")

        plt.title("Original Prediction + Sea Mask Overlay — unmasked")
        plt.tight_layout()
        plt.savefig("predictionMasked_raw_plus_sea_mask_overlay.png", dpi=300)
        plt.close()
        """

        # applying filters per pixel
        print("Applying full dBZ + prediction + distance + sea mask filtering...")

        max_dBZ_threshold = 38  # dBZ for ~8000 White Pelicans; T-matrix RCS (calculated in dBZ_to_reflectivity_n.py)
        min_dBZ_threshold = 0  # dBZ for ~1 White Pelican
        max_distance = 50000  # in meters

        for data in birds_data:
            dBZ = data.pop('dBZ')  # Original dBZ values
            prediction = data['pred_0_1']
            euclidean_dist = data['euclidean_distance']

            # Start with valid dBZ mask
            valid_dBZ_mask = (dBZ >= min_dBZ_threshold) & (dBZ < max_dBZ_threshold)

            # Combine all conditions
            final_mask = (
                    ~valid_dBZ_mask |  # Outside dBZ threshold
                    (prediction == 0) |  # Model did not predict a bird
                    (euclidean_dist > max_distance) |  # Too far
                    (corrected_sea_mask)  # Over the sea
            )

            # Apply all filters at once
            filtered_dBZ = dBZ.copy()
            filtered_dBZ[final_mask] = 256  # Sentinel value for masked pixels

            # Save result
            data['filtered_prediction_dBZ_0'] = filtered_dBZ

        gc.collect()
        print("Combined filtering per pixel complete.")

        # -------------------------------
        # Convert dBZ to reflectivity
        # -------------------------------
        """        
        In radar meteorology reflectivity factors z (or Z in dBZ, note capital notation)) are the conventional unit (for its useful property of being
        independent of radar wavelength in the case of small scatterers like precipitation (Doviak and Zrnić 1993), for larger
        animals like birds a more useful unit is reflectivity η (Dokter et al. 2011, Chilson et al. 2012b), which is more
        directly proportional to aerial animal density (see  caption  Fig. 1 for conversions) in the bioRad paper.
        """
        print("Calculate reflectivity...")
        for data in birds_data:
            fp.dBZ_to_reflectivity(data)

        gc.collect()

        # ----------------------
        #  Filters per cluster
        # ----------------------
        print("Masking out clusters with high reflectivity density and by size and width...")

        # //// Set thresholds by month ////
        # -----------------------------------
        if month == '09':
            THRESHOLD_RATIO = 30670  # high ratio = cloud
            THRESHOLD_SIZE = 488
            THRESHOLD_WIDTH = 29
            THRESHOLD_REF_SUM = 4150000
            THRESHOLD_REF_SUM_COMBINE = 3650000
            THRESHOLD_WIDTH_COMBINE = 18
        else:
            THRESHOLD_RATIO = 10170  # high ratio = cloud
            THRESHOLD_SIZE = 122
            THRESHOLD_WIDTH = 13
            THRESHOLD_REF_SUM = 341900
            THRESHOLD_REF_SUM_COMBINE = None  # not used in non-September months
            THRESHOLD_WIDTH_COMBINE = None

        for data in birds_data:
            refl = data.get("prediction_reflectivity_eta_0")
            if refl is None:
                continue

            mask = (refl != 0)
            labeled, num_features = label(mask)

            for comp_id in range(1, num_features + 1):
                row_idx, col_idx = np.where(labeled == comp_id)
                size = len(row_idx)
                if size == 0:
                    continue

                reflectivity_sum = refl[row_idx, col_idx].sum()
                reflectivity_ratio = reflectivity_sum / size
                horizontal_width = fp.compute_max_horizontal_width(row_idx, col_idx)

                if (reflectivity_ratio >= THRESHOLD_RATIO
                        or size >= THRESHOLD_SIZE
                        or horizontal_width >= THRESHOLD_WIDTH
                        or reflectivity_sum >= THRESHOLD_REF_SUM
                        or (THRESHOLD_REF_SUM_COMBINE is not None
                            and THRESHOLD_WIDTH_COMBINE is not None
                            and reflectivity_sum >= THRESHOLD_REF_SUM_COMBINE
                            and horizontal_width >= THRESHOLD_WIDTH_COMBINE)):
                    refl[row_idx, col_idx] = 0  # Mark as invalid

        print("Clustering filter complete.")

        """ debugging
        subset_birds_data = [
            d for d in birds_data
            if d.get('date') == '20240909' and d.get('time') == '090002'
        ]
        """

        # ---------------------------
        # Extracting height ranges
        # ---------------------------
        # upload birds_h5_files - a list of files with only birds prediction:
        file_name = f"{SITE}_{year_month}.json"
        output_dir = fr'MY\PATH\TO\bird_h5_files_json/elev_{elev_predict}/'
        output_path = os.path.join(output_dir, file_name)

        # Load the JSON data
        with open(output_path, 'r') as f:
            birds_h5_files = json.load(f)

        # Path was change from 'M:/Beit_dagan/H5/2024/08/14\\0000ISR-PPIVol-20240814-081503-0a0a.hdf' so I update it here: may not alase be neseccery
        # birds_h5_files = [path.replace('M:/', 'K:/') for path in birds_h5_files]

        print("Calculate height...")
        start = time.time()

        '''For each elevation and distance, the max height is practically fixed'''
        birds_data = fp.run_height_range_in_chunks(birds_h5_files, birds_data, radar_height, elev_predict,
                                                   chunk_size=50)

        end = time.time()
        print(f"run_height_range_in_chunks took {end - start:.2f} seconds")

        # Memory cleanup
        del birds_h5_files

        gc.collect()

        """ debugging
                subset_birds_data = [
                    d for d in birds_data
                    if d.get('date') == '20240909' and d.get('time') == '090002'
                ]
                """
        # ---------------------------------------------------------
        # Compute bird count
        # ---------------------------------------------------------
        print(f'Compute bird count...')

        # Extract date range
        start = time.time()

        for index, row in mean_RCSs.iterrows():
            date_range = row['dates range (central 90%)']
            start_str, end_str = date_range.split(" - ")
            mean_RCSs.loc[index, 'start_date'] = datetime.strptime(start_str, "%d.%m")  # Convert start date
            mean_RCSs.loc[index, 'end_date'] = datetime.strptime(end_str, "%d.%m")  # Convert end date

        # Extract month and day from the date
        for data in birds_data:
            # data = birds_data[0]
            date_obj = datetime.strptime(data['date'], "%Y%m%d")  # Convert to datetime object
            month_day = datetime.strptime(f"{date_obj.day}.{date_obj.month}", "%d.%m")  # Format as 'DD.MM'

            # Find the correct RCS value from df_mean_RCSs
            matching_row = mean_RCSs[(mean_RCSs["start_date"] <= month_day) & (mean_RCSs["end_date"] >= month_day)]
            RCS_Tmatix_value = matching_row['RCS_Tmatrix'].values[0]   # RCS_Tmatrix - name from the general RCS table; Weighted_Mean_RCS - name from the weighted RCS table

            arr = np.asarray(data['prediction_reflectivity_eta_0'])
            bc_tmatrix = np.empty_like(arr, dtype=np.float32)

            np.divide(arr, RCS_Tmatix_value, out=bc_tmatrix, dtype=np.float32)

            data['bird_count_0_Tmatrix'] = bc_tmatrix
            gc.collect()

        end = time.time()
        print(f"Compute bird count took {end - start:.2f} seconds")
        gc.collect()

        # ONLY FOR DAGAN 2014
        """ 
        for data in birds_data:
            date = pd.to_datetime(data['date'])
            rcs_row = weighted_rcs_by_date_oct_2014[
                weighted_rcs_by_date_oct_2014['Date'] == date]  # TODO: change when finishing with dagan 2014 october

            if not rcs_row.empty:
                RCS_Tmatix = rcs_row['weighted_RCS_Tmatrix'].values[0]
                RCS_wipl = rcs_row['weighted_RCS_WIPLD'].values[0]

                eta = data['filtered_overlap_heights_prediction_eta_0']
                data['filtered_overlap_heights_bird_count_0_Tmatrix'] = eta/RCS_Tmatix
                data['filtered_overlap_heights_bird_count_0_wipl'] = eta/RCS_wipl

                data['filtered_overlap_heights_eta_sum_0'] = eta.sum()
            else:
                print(f"No RCS data found for date {date}")
        """

        # ------------------------------------------------------------------------------
        # Checkpoint_3: save list of dictionaries - pixle level, only relevant arrays
        # ------------------------------------------------------------------------------
        start = time.time()

        keys_to_keep = ['date', 'time', 'angle', 'lon_grid', 'lat_grid', 'bird_count_0_Tmatrix',
                        'prediction_reflectivity_eta_0', 'min_height', 'max_height',
                        'euclidean_distance']  # 'height_range',

        birds_data_light = []

        for scan in birds_data:
            scan_copy = {k: v for k, v in scan.items() if k in keys_to_keep}
            birds_data_light.append(scan_copy)

        file_name = f"birds_data_{SITE}_{year_month}.joblib"
        output_dir = fr'MY\PATH\TO\bird_data_filtered_prediction_Checkpoint_3_pkl/elev_{elev_predict}/'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, file_name)

        joblib.dump(birds_data_light, output_path)  # save

        end = time.time()
        print(f"saving Checkpoint_3 took {end - start:.2f} seconds")

        gc.collect()

        '''
        # -------------------------------------------------------
        # Checkpoint_3: save list of dictionaries - pixle level, all variables
        # -------------------------------------------------------
        start = time.time()

        file_name = f"birds_data_{SITE}_{year_month}.joblib"
        output_dir = fr'MY\PATH\TO\bird_data_filtered_prediction_Checkpoint_3_pkl/elev_{elev_predict}/'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, file_name)

        joblib.dump(birds_data, output_path)  # save

        end = time.time()
        print(f"saving Checkpoint_3 took {end - start:.2f} seconds")
        '''
        # Upload the data:
        """
        birds_data = joblib.load(output_path)
        """

        # ----------------------------------------------------------------------------------
        # run clustering on each scan to get pixel groupings - assuming flocks (cluster IDs).
        # -----------------------------------------------------------------------------------
        print(f'create clusters for flocks...')

        start = time.time()

        clustered_data = []  # This will hold tuples of (scan_dict, cluster_mask)
        cluster_meta = []  # one row per cluster with metrics

        for scan in birds_data:
            birdcount = scan.get("bird_count_0_Tmatrix")
            refl = scan.get("prediction_reflectivity_eta_0")
            mask = refl > 0  # consider clusters only for values greater than 0

            # Label clusters
            # structure = np.ones((3, 3), dtype=int)  # includes diagonal toching pixels as part of the cluster
            # labeled_array, num_clusters = label(mask, structure=structure)
            labeled_array, num_clusters = label(mask)
            # - the label function, performs connected-component labeling. It scans the 2D array and groups adjacent True pixels into distinct components.
            #   label(mask) finds all connected regions of True pixels in mask. By default, it uses 8-connectivity in 2D if no structure parameter is given (depending on SciPy version). This means pixels are considered connected if they share an edge or a corner.
            # - labeled_array is an integer array the same shape as mask.
            #   Each distinct connected region of True pixels has a unique label (1..num_features).
            # - num_clusters is the total number of connected regions found.

            for cluster_id in range(1, num_clusters + 1):
                cluster_mask = labeled_array == cluster_id  # Create a boolean mask for all pixels that belong to the current cluster, so it isolates clusters of the same scan from each other
                row_idx, col_idx = np.where(cluster_mask)
                size = len(row_idx)

                if size == 0:
                    continue

                reflectivity_sum = refl[row_idx, col_idx].sum()
                reflectivity_ratio = reflectivity_sum / size

                horizontal_width = compute_max_horizontal_width(row_idx, col_idx)

                clustered_data.append((scan, cluster_mask, cluster_id))

                # store one meta row per cluster; types match gaussian_weighted_cluster outputs
                cluster_meta.append({
                    "date": datetime.strptime(scan["date"], "%Y%m%d").date(),
                    "time": datetime.strptime(scan["time"], "%H%M%S").time(),
                    "angle": scan["angle"],
                    "cluster_id": cluster_id,
                    "cluster_size": size,
                    "cluster_width": horizontal_width,
                    "cluster_reflectivity_sum": reflectivity_sum,
                    "cluster_reflectivity_ratio": reflectivity_ratio,
                })

        all_cluster_meta.extend(cluster_meta)

        end = time.time()

        print(f"create clusters for flocks took {end - start:.2f} seconds")

        """ debugging
        subset_birds_data = [
            d for d in birds_data
            if d.get('date') == '20240909' and d.get('time') == '090002'
        ]
        
         subset_clustered_data = [
            d for d in birds_data
            if d.get('date') == '20240909' and d.get('time') == '090002'
        ]
        """

        del birds_data
        gc.collect()

        # ----------------------------------------------------------
        # run PDF (Probability Density Function) on cluster level
        # ------------------------------------------------------------
        # missing_min_height = [i for i, d in enumerate(birds_data) if 'angle' not in d]  # min_height
        # print(f"Dictionaries missing 'min_height': {missing_min_height}")
        # for i in missing_min_height:
        #       print(f"Index: {i}, Date: {birds_data[i].get('date')}, Time: {birds_data[i].get('time')}")

        print(f'Compute gaussian_weighted_cluster...')

        start = time.time()

        cluster_results = []  # will hold a list of dictioneries where one dictionary per height-bin × pixel in every retained cluster in every scan.

        for scan, mask, cluster_id in clustered_data:
            cluster_records = fp.gaussian_weighted_cluster(scan, mask, cluster_id)
            cluster_results.extend(
                cluster_records)  # adds all the inner dictionaries (not the list itself) from cluster_records into a growing master list.

        end = time.time()
        print(f"Compute gaussian_weighted_cluster took {end - start:.2f} seconds")
        gc.collect()
        # convert the entire list into df where each row is one height bin in one pixel in one cluster.
        final_df = pd.DataFrame(cluster_results)

        # ---------------------------------------------------------------------------------
        # Checkpoint_4: save the final list of dictionaries - cluster_results
        # ---------------------------------------------------------------------------------
        print(f'save Checkpoint_4 - data as pixels but clusters associated...')

        file_name = f"birds_data_{SITE}_{year_month}.joblib"
        output_dir = fr'MY\PATH\TO\bird_data_filtered_prediction_Checkpoint_4_pkl/elev_{elev_predict}/'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, file_name)

        start = time.time()

        # Save the list of dictionaries as a joblib file
        joblib.dump(cluster_results, output_path, compress=3)

        end = time.time()
        print(f"save Checkpoint_4 took {end - start:.2f} seconds")
        gc.collect()

        print("Cleaning memory...")
        del clustered_data
        del cluster_results
        del final_df

        gc.collect()

    # ==================================================
    # After all elevations are processed for this month:
    # Combine and clean duplicates across angles
    # ==================================================
    print(f"Combining and cleaning all elevations for {year_month}...")

    pattern = (
        fr"MY\PATH\TO\bird_data_filtered_prediction_Checkpoint_4_pkl"
        fr"\elev_*\birds_data_{SITE}_{year_month}.joblib"
    )
    all_files = glob.glob(pattern)

    if not all_files:
        print(f"No pixel-level Checkpoint_4 joblib files found for {year_month}, skipping.")
        continue
    else:
        all_data = []
        for f in all_files:
            print(f"Loading {os.path.basename(f)} ...")
            data = joblib.load(f)  # Each file is a list of dictionaries
            all_data.extend(data)

        print(f"Total records combined before cleaning: {len(all_data):,}")

    # Convert to DataFrame for deduplication
    df_all = pd.DataFrame(all_data)

    # Prepare voxel ID (unique 3D location + time)
    df_all["lon_round"] = df_all["long"].round(5)
    df_all["lat_round"] = df_all["lat"].round(5)
    df_all["voxel_id"] = (
            df_all["date"].astype(str)
            + "_" + df_all["time"].astype(str)
            + "_" + df_all["lon_round"].astype(str)
            + "_" + df_all["lat_round"].astype(str)
            + "_" + df_all["height_bin"].astype(str)
    )

    # Clean duplicates (keep the voxel with the highest bird_count)
    df_clean = (
        df_all.sort_values(by=["bird_count"], ascending=False)
        .drop_duplicates(subset="voxel_id", keep="first")
    )

    print(f"Before cleaning duplicates: {len(df_all):,}")
    print(f"After cleaning duplicates: {len(df_clean):,}")
    print(f"number of cleaning pixels: {len(df_all) - len(df_clean):,}")

    # -------------------------------------------------------
    # Create cluster-level aggregation after cleaning
    # -------------------------------------------------------
    print(f"Aggregating cleaned data to cluster level...")

    # Group by unique flock identifiers
    grouped = df_clean.groupby(['date', 'time', 'angle', 'cluster_id', 'height_bin'])

    aggregated = grouped.agg({
        'reflectivity': 'sum',
        'bird_count': 'sum',
        'distance_from_radar': 'mean',
        'cell_height_range': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    }).reset_index()

    # Compute cluster centroids from all pixel coordinates
    centroid_data = grouped.apply(
        lambda df: pd.Series({
            'centroid_long': MultiPoint(list(zip(df['long'], df['lat']))).centroid.x,
            'centroid_lat': MultiPoint(list(zip(df['long'], df['lat']))).centroid.y
        }),
        include_groups=False
    ).reset_index()

    cluster_summary_by_height = pd.merge(
        aggregated,
        centroid_data,
        on=['date', 'time', 'angle', 'cluster_id', 'height_bin']
    )

    meta_df = pd.DataFrame(all_cluster_meta)

    cluster_summary_by_height = cluster_summary_by_height.merge(
        meta_df[[
            "date", "time", "angle", "cluster_id",
            "cluster_size", "cluster_width",
            "cluster_reflectivity_sum", "cluster_reflectivity_ratio"
        ]],
        on=["date", "time", "angle", "cluster_id"],
        how="left"
    )

    # ---------------------------------------------------
    # Export the cluster-level summary
    # ---------------------------------------------------
    print(f'Exporting DF...')

    gc.collect()

    file_name = f'{SITE}_{year_month}.csv'
    directory_path = r'MY\PATH\TO\1st_clean_aggregated_csv'
    file_path = os.path.join(directory_path, file_name)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    print("Exporting final database...")
    cluster_summary_by_height.to_csv(file_path, index=False)

    print("Cleaning memory...")

    # Memory cleanup
    del all_files
    del df_all
    del all_data
    del df_clean
    del grouped
    del aggregated
    del centroid_data
    del cluster_summary_by_height
    del meta_df

    gc.collect()  # Force garbage collection

    print(f"Cluster-level file saved: {file_path}")
