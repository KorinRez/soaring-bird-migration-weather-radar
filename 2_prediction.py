#########################################################################################################
#
#                                   Setting the environment
#
#########################################################################################################
import numpy as np
import glob
import os
import sys
import copy
import pickle
from datetime import datetime

# from Flight_properties_functions import *

# PREDICT NEW DATA VRDA
# This is the path to NET-flocks-detection
sys.path.append(
    r'MY:\PATH\TO\UNET-flocks-detection_functions')

from create_previous_images import create_early_image_2
from generators import image_generator
from unet_model import unet

#########################################################################################################
#
#                                        DEFINE PARAMETERS!
# TODO: the only place in the script that needs changes!!!!!
#
#########################################################################################################
SITE = 'dagan'
ELEVATIONS = [1, 2, 3, 4, 5, 6]  # 1 is 0.5 or 0.3 in beit dagan. it represent dataset1 , 6, 7, 8
year = '2025'
months = ['10', '11']  # '03', '04', '05', '08','09', , '10', '11'
# month = '09'
# year_month = '2014_september'
months_str = ['october', 'november']  # 'march', 'april', 'may', 'august','september','october', 'november'


# year_month = '2024_september'

#########################################################################################################
#
#                                    PREDICT NEW DATA - VRDH
#
#########################################################################################################
def main():
    try:
        print("Running script 2_prediction_beit_dagan.py")

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
                # elev_predict = ELEVATIONS[0]  # for debugging
                print(f'processing elevation {elev_predict}')

                # UPLOAD TIFF FILES
                # ------------------
                test_files_n = glob.glob(
                    f'L:/nir_lab_11/Korin_PhD/HDF_to_PPI/ppi_vrad/{SITE}/{year}/{month}/*/*/elev_{elev_predict}/*.tiff')

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
                test_files >= 24
                Less than 64 pictures make a problem in the rest of the for loop code. For now, it's mean I need to run the for loop
                # until the shape of x_i.shape[0] is less than 32, and because the generator choose pic randomly, and it need 3
                # consecutive pic,the number change. 32 break the code. And remember to run the right and left variable to restore
                # them to zero!!!!! before rerunning the code
                '''
                batch_size = 32
                test_steps = len(test_files) // batch_size
                x_1 = batch_size * test_steps
                x_test = np.zeros((x_1, 256, 256, 9))
                img_data = []
                right = 0
                left = 0

                del test_files  # memory cleaning

                print("Running the generator...")
                # here is the problem when we have less than 64 pictures
                for x_i, img_d in test_generator:
                    # When starting the loop, first the generator work and at the end of its process x_i = batch_x and y_i = batch_y
                    left = right
                    right += x_i.shape[0]
                    if right >= x_1:
                        break
                    # The statement left:right add all the pictures we processed at the generator in each loop
                    x_test[left:right, :, :, :] = x_i
                    img_data[left:right] = img_d

                x_test = x_test[:left, :, :, :]
                img_data = img_data[:left]  # a dictionary with date, time and x_test values

                print('x_test', x_test.shape)
                print('img_data length', len(img_data))
                print('img_data', img_data[0]['img'].shape)

                #  EXECUTING THE PREDICTION
                # ---------------------------
                print("Running the prediction model...")
                predict = model.predict(x_test)
                print(predict.shape)

                del x_test  # memory cleaning

                # SAVE THE PREDICTION
                # ---------------------
                # Here the prediction is ** BETWEEN 0 TO 1 ** because in the Bird_intensity we resize the array, so I want it to average
                # these values and not the 0 or 1 values (assuming that's what the resize function does).
                img_data_predict = copy.deepcopy(img_data)

                # test - checking if the img (here 256X256) is flipped like in the 4 script:
                """
                img_data_predict_sorted = sorted(img_data_predict, key=lambda x: (x["date"], x["time"]))
            
                prediction_test = img_data_predict_sorted[65]
                img_test = prediction_test['img']

                # Create a base plot of the image
                from matplotlib import pyplot as plt
                
                plt.figure(figsize=(8, 8))
                plt.imshow(img_test, cmap="viridis")  # , origin="lower"; Display the image with the viridis colormap
                plt.savefig("unpliped.png")
                plt.close()
                # and the answer is yes!
                """

                ## Changing the values of img in img data (a dictionary) from pixels values to prediction values
                for i, data in enumerate(img_data_predict):
                    img_data_predict[i]['img'] = predict[i]

                # Checking the inside values are equal - debugging
                # img_predict_values = [i['img'] for i in img_data_predict]
                # print(np.allclose(predict[:, :, :, :], img_predict_values))

                ## Save the prediction to a folder
                file_name = f"{SITE}_{year_month}_predict.pkl"
                # file_name = f"{SITE}_{year_month}_part1_predict_200to256_newModel.pkl"  # did it when the data was too large for the systen memory
                dir_pred = fr'L:\nir_lab_11\Korin_PhD\analyzed_radar_data_beit_dagan\prediction\elev_{elev_predict}'

                # Check if the directory exists; if not, create it
                if not os.path.exists(dir_pred):
                    os.makedirs(dir_pred)
                file_path = os.path.join(dir_pred, file_name)

                with open(file_path, 'wb') as f:
                    pickle.dump(img_data_predict, f)

                # SAVE THE img
                # -------------------------------
                ## Save the img to a folder
                file_name = f"{SITE}_{year_month}_img_pxl.pkl"
                # file_name = f"{SITE}_{year_month}_part1_predict_200to256_newModel.pkl"  # did it when the data was too large for the systen memory
                dir_pred = fr'L:\nir_lab_11\Korin_PhD\analyzed_radar_data_beit_dagan\img_pxl\elev_{elev_predict}'

                # Check if the directory exists; if not, create it
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
