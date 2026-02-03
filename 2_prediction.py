# ---------------------------------- Prediction of soaring flocks ------------------------------------
"""
Code adapted from: Inbal Schekler's UNET-flocks-detection repository
Original source: https://github.com/Inbal-Schekler/UNET-flocks-detection/tree/main
"""

# ----------------------------------------------------------------------------------------------------
#
#                                   Setting the environment
#
# ----------------------------------------------------------------------------------------------------
import numpy as np
import glob
import os
import sys
import copy
import pickle
from datetime import datetime

# TODO: modify as needed - this is the path to UNET-flocks-detection functions by Schekler et al. (2023) Methods in Ecology and Evolution
sys.path.append(
    r'MY:\PATH\TO\UNET-flocks-detection_functions')

from create_previous_images import create_early_image_2
from generators import image_generator
from unet_model import unet

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
months_str =  ['august', 'september','october', 'november'] 

# ----------------------------------------------------------------------------------------------------
#
#                                    PREDICTION
#
# ----------------------------------------------------------------------------------------------------

def main():
    try:
        print("Running script 2_prediction.py")

        # SETTING THE MODEL
        # ------------------
        model = unet()
        # model.summary()

        ## Loading the weights of the best epoch
        model.load_weights(
            r"my_best_model_2_prev_inc_noth.epoch17-loss0.01.hdf5")

        for m, m_str in zip(months, months_str):
            month = m
            year_month = f'{year}_{m_str}'
            print(month, year_month)

            for i, elev in enumerate(ELEVATIONS):
                elev_predict = ELEVATIONS[i]
                print(f'processing elevation {elev_predict}')

                # UPLOAD TIFF FILES (the PPI images
                # ----------------------------------
                # TODO: Modify as needed - path was defined in script 1_hdf_to_ppi
                test_files_n = glob.glob(
                    f'MY:/PATH/TO_PPI_IMAGES/HDF_to_PPI/ppi_vrad/{SITE}/{year}/{month}/*/*/elev_{elev_predict}/*.tiff')

                if not test_files_n or len(test_files_n) < 32:
                    print(
                        f"Not enough files found for month {month}, elevation {elev_predict} (found {len(test_files_n)}). Skipping...")
                    continue

                print('file format:', os.path.basename(test_files_n[0]))

                test_files = sorted(test_files_n, key=lambda file: datetime.strptime(
                    os.path.basename(file).split('-')[2] + ' ' + os.path.basename(file).split('-')[3], '%Y%m%d %H%M%S'))
                print("length of test files", len(test_files))

                # CREATING THE GENERATOR
                # ------------------------
                num_past = 2
                minuts = 7
                batch_size = 32
                test_generator = image_generator(test_files, num_past=num_past, minuts=minuts, batch_size=batch_size)

                # EXECUTING THE GENERATOR:
                # ------------------------
                '''
                Creating array of the test files for predictions
                requirement:
                test_files > 32
                '''
                batch_size = 32
                test_steps = len(test_files) // batch_size
                x_1 = batch_size * test_steps
                x_test = np.zeros((x_1, 256, 256, 9))
                img_data = []
                right = 0
                left = 0

                del test_files  # Free up memory. Optional.

                print("Running the generator...")
                for x_i, img_d in test_generator:
                    left = right
                    right += x_i.shape[0]
                    if right >= x_1:
                        break
                    x_test[left:right, :, :, :] = x_i
                    img_data[left:right] = img_d

                x_test = x_test[:left, :, :, :]
                img_data = img_data[:left]  # a dictionary include date, time and x_test values

                print('x_test', x_test.shape)
                print('img_data length', len(img_data))
                print('img_data', img_data[0]['img'].shape)

                #  EXECUTING THE PREDICTION
                # ---------------------------
                print("Running the prediction model...")
                predict = model.predict(x_test)
                print(predict.shape)

                del x_test  # Free up memory. Optional.

                # SAVE THE PREDICTION
                # ---------------------
                # Here the prediction values are between 0 and 1. In script 3_extracting_ppi_metadata, we resize the array for more spatial details, and then convert values to 0 or 1.
                img_data_predict = copy.deepcopy(img_data)

                # --- Debugging: Check if the image (256x256) is flipped (not not North-South oriented). Optional.---
                #img_data_predict_sorted = sorted(img_data_predict, key=lambda x: (x["date"], x["time"]))
            
                #prediction_test = img_data_predict_sorted[65]
                #img_test = prediction_test['img']

                # Create a base plot of the image
                #from matplotlib import pyplot as plt
                
                #plt.figure(figsize=(8, 8))
                #plt.imshow(img_test, cmap="viridis")  # , origin="lower"; Display the image with the viridis colormap
                #plt.savefig("unpliped.png")
                #plt.close()
                
                ## Changing the values of img in img data (a dictionary) from pixels values to prediction values
                for i, data in enumerate(img_data_predict):
                    img_data_predict[i]['img'] = predict[i]


                ## Save the prediction to a folder
                file_name = f"{SITE}_{year_month}_predict.pkl"
                dir_pred = fr'MY:\PATH\TO\prediction\elev_{elev_predict}'

                # Check if the directory exists; if not, create it
                if not os.path.exists(dir_pred):
                    os.makedirs(dir_pred)
                file_path = os.path.join(dir_pred, file_name)

                with open(file_path, 'wb') as f:
                    pickle.dump(img_data_predict, f)

                # Save the img
                # -------------------------------
                ## Save the img to a folder as an array of pixel
                file_name = f"{SITE}_{year_month}_img_pxl.pkl"
                dir_pred = fr'MY:\PATH\TO\img_pxl\elev_{elev_predict}'

                # Check if the directory exists; if not, create it.
                if not os.path.exists(dir_pred):
                    os.makedirs(dir_pred)
                file_path = os.path.join(dir_pred, file_name)

                with open(file_path, 'wb') as f:
                    pickle.dump(img_data, f)

                del img_data_predict
                del img_data
                del predict
                del test_files_n
                del img_d
                del x_i
                print(f'Prediction of elevation {elev_predict} of {SITE} {year} {month} is done')

        print(f'Prediction of month {year_month}  is done')

        print(f'script 2 of {SITE} {year} {month} is finished')

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)  # Exit with a non-zero status code to indicate an error


if __name__ == "__main__":
    main()
