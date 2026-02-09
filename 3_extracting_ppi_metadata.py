# ----------------------------------------------------------------------------------------------------
#
#                                   Setting the environment
#
# ----------------------------------------------------------------------------------------------------

import numpy as np
import os
import pickle
import json
import time
import gc
import joblib
import cv2

import Flight_properties_functions as fp

# ----------------------------------------------------------------------------------------------------
#
#                                        DEFINE PARAMETERS
#
## ----------------------------------------------------------------------------------------------------

# TODO: Adjust the parameters below according to your needs

SITE = 'name_of_your_site'
ELEVATIONS = [1, 2, 3, 4, 5, 6]  
year = '2025'
months = ['08', '09', '10', '11']
months_str = ['august', 'september','october', 'november'] 
targetShape = (400, 400)

# ----------------------------------------------------------------------------------------------------
#
#                                           MAIN
#
# ----------------------------------------------------------------------------------------------------

# Iterate the month
# ===============================================
for m, m_str in zip(months, months_str):
    month = m
    year_month = f'{year}_{m_str}'
    print(month, year_month)

    # import sorted_files (h5 files that are filtered for day-time data and above certain size - done in script 1_hdf_to_ppi.py)
    # --------------------------------------------------------------------------------------------------------------------------
    # TODO: Modify as needed - path was defined in script 1_hdf_to_ppi
    file_name = f"{SITE}_{year_month}_h5.pkl"
    file_path = fr'MY:\PATH\TO\daytime_h5_list\{file_name}'

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping...")
        continue

    with open(file_path, 'rb') as f:
        sorted_files = pickle.load(f)

    # Iterate the elevation
    # ============================================
    for i, elev in enumerate(ELEVATIONS):
        elev_predict = ELEVATIONS[i]
        print(f'processing elevation {elev_predict}')

        # Load the bird_prediction_dict from a pickle file (done in script 2_prediction.py)
        # ----------------------------------------------------------------------------------
        # TODO: Modify as needed - path was defined in script 2_prediction.py
        file_name = f"{SITE}_{year_month}_predict.pkl"
        file_path = fr'MY:\PATH\TO\prediction\elev_{elev_predict}\{file_name}'

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}. Skipping...")
            continue

        with open(file_path, 'rb') as f:
            bird_prediction = pickle.load(f)

        bird_prediction = fp.sort_by_date_time(bird_prediction)

        # Remove duplicates:
        duplicates = set()

        for idx in range(len(bird_prediction)):
            for j in range(idx + 1, len(bird_prediction)):
                if (
                        bird_prediction[idx]["date"] != bird_prediction[j]["date"]
                        or bird_prediction[idx]["time"] != bird_prediction[j]["time"]
                ):
                    break  # Exit the inner loop when date or time changes
                if np.array_equal(bird_prediction[idx]["img"], bird_prediction[j]["img"]):
                    # duplicates.add(idx)
                    duplicates.add(j)

        print("Duplicate indices of bird_prediction:", len(duplicates))

        # Create a new list without duplicates:
        bird_prediction = [d for i, d in enumerate(bird_prediction) if i not in duplicates]

        
        # Prepare a list of files with only birds' prediction to extract dBZ values
        # -------------------------------------------------------------------------
        birds_h5_files = fp.loadRelevantFiles(bird_prediction, sorted_files)

        # Save to a JSON file
        # TODO: modify path as needed
        file_name = f"{SITE}_{year_month}.json"
        output_dir = fr'MY\PATH\TO\bird_h5_files_json/elev_{elev_predict}/'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, file_name)

        with open(output_path, 'w') as f:
            json.dump(birds_h5_files, f)

        
        # Extracting dBZ, latitude, and longitude from PPI
        # -----------------------------------------------
        ### Note: This is a computationally intensive task and execution time may vary depending on the amount of files (4 files takes avout 22 seconds to process.
        print(f'extracting PPI metadata for {SITE}_{year_month} elev_{elev_predict}')

        # Parameters
        grid_size = 250
        range_max = 50000
        array_size = 400
        chunk_size = 200  
        scan_num = elev_predict - 1

        proj4_string = fp.extract_proj4_from_first_file(birds_h5_files[0], grid_size=250, range_max=50000)

        print("Starting PPI extraction...")
        start = time.time()

        ppi_birds_data = fp.process_PPI_files_in_chunks(
            file_paths=birds_h5_files,
            scan_num=scan_num,
            proj4_string=proj4_string,
            grid_size=grid_size,
            range_max=range_max,
            array_size=array_size,
            chunk_size=chunk_size
        )

        end = time.time()

        print(f"\n Finished extracting {len(ppi_birds_data)} files. Took {end - start:.2f} seconds, now saving...")

        gc.collect()

        ppi_birds_data = fp.sort_by_date_time(ppi_birds_data)

  
        # Resize prediction array to fit dBZ extracted data array (400x400)
        # -----------------------------------------------------------------
        print("Resize prediction array")
      
        for pred_dict in bird_prediction:
            pred_array = pred_dict["img"]  # Key for the 256x256x1 array
            resized_pred = cv2.resize(pred_array.squeeze(), targetShape, interpolation=cv2.INTER_LINEAR)
            pred_dict["prediction_resized"] = resized_pred
            del pred_dict["img"]

        
        # Create distance array as in the h5 files
        # -------------------------------------------
        print("Calculate distance...")

        for dictionary in ppi_birds_data:
            fp.Cartesian_distance_to_Euclidean(dictionary)
            del dictionary['x_coords']
            del dictionary['y_coords']

        gc.collect()

        # Create one list of dictionaries that include ppi metadata and prediction values
        # ---------------------------------------------------------------------------------
        # merge dictionaries
        print("marge dictionaries...")

        birds_data = fp.merge_dictionaries(ppi_birds_data, bird_prediction)

        del ppi_birds_data
        del bird_prediction
        gc.collect()

        
        # Changing the predictions from a range between 0 and 1 to a binary prediction of 0 or 1
        # ---------------------------------------------------------------------------------------
        print("Changing the predictions from a range between 0 to 1 to a binary prediction of 0 or 1...")

        for data in birds_data:
            pred_0_1 = data.pop("prediction_resized")
            pred_0_1[pred_0_1 >= 0.5] = 1
            pred_0_1[pred_0_1 < 0.5] = 0
            data["pred_0_1"] = pred_0_1  

        # Saving ppi metadata (.joblib file)
        # -----------------------------------
        file_name = f"PPI_metadata_{SITE}_{year_month}.joblib"
        output_dir = fr'MY:\PATH\TO\PPI_metadata/elev_{elev_predict}/'
        output_path = os.path.join(output_dir, file_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the list of dictionaries as a joblib file
        joblib.dump(birds_data, output_path)  

        print(f"birds_data.joblib was saved for {SITE}_{year_month}_elev_{elev_predict}")

    del sorted_files
    gc.collect()
