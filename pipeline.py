import datetime
import os
import cv2
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.curve_fitting_utils import sample_equidist
from utils.ef_functions import ef_coefs
from utils.img_preproc_utils import ImgProcessor, ContourGSN, transform_gsn

folder = 'data/landmark_placement/img'

imgs = os.listdir(folder)
data_coefs = []

for file_name in imgs:

    img = ImgProcessor.read(folder, file_name)
    print('Processing image' + file_name)
    print("- Number of Pixels: " + str(img.size))
    print("- Shape/Dimensions: " + str(img.shape))

    img_new = img.resize()
    img_gray, img_blur, img_open, gsn_mask = img_new.mask()

    img_contour = ContourGSN.find_contour(img_new, gsn_mask)
    img_contour.segment_gsn()

    res = transform_gsn(img_contour.points_flip[0], img_contour.points_flip[1], img_contour.gsn_curve_flip)

    # Save results: each img into separate folder with img id name
    # inside: cropped resized img, gsn_contour before transform, gsn contour after transform
    new_folder = os.path.join(os.path.split(folder)[0], 'segmented_curves')
    new_subfolder = os.path.splitext(file_name)[0]
    if not os.path.exists(os.path.join(new_folder, new_subfolder)):
        os.makedirs(os.path.join(new_folder, new_subfolder))
    cv2.imwrite(os.path.join(new_folder, new_subfolder, os.path.splitext(file_name)[0] + '_gsn_img.png'),
                img_contour.img_flip)
    np.savetxt(os.path.join(new_folder, new_subfolder, 'gsn_contour.csv'), img_contour.gsn_contour_flip, delimiter=",")
    np.savetxt(os.path.join(new_folder, new_subfolder, 'gsn_curve.csv'), img_contour.gsn_curve_flip, delimiter=",")
    np.savetxt(os.path.join(new_folder, new_subfolder, 'gsn_curve_registered.csv'), res, delimiter=",")

    fig, ax = plt.subplots(5, 1, figsize=(10, 10))
    ax[0].imshow(cv2.cvtColor(img_new.img, cv2.COLOR_BGR2RGB), cmap=plt.cm.Spectral)
    ax[1].imshow(img_blur, cmap='gray')
    ax[2].imshow(gsn_mask)
    ax[3].imshow(cv2.cvtColor(img_new.img, cv2.COLOR_BGR2RGB), cmap=plt.cm.Spectral)
    for contour in img_contour.contours:
        ax[3].plot(contour[:, 1], contour[:, 0], linewidth=1.2)
    ax[3].plot(img_contour.points[0][1], img_contour.points[0][0], 'xr')
    ax[3].plot(img_contour.points[1][1], img_contour.points[1][0], 'xr')
    ax[4].plot(res[:, 1], res[:, 0], '-r', linewidth=1.2)
    ax[4].set_box_aspect(gsn_mask.shape[0] / gsn_mask.shape[1])
    ax[4].set_ylim(ymin=0)
    ax[4].axis('equal')
    plt.savefig(os.path.join(new_folder, new_subfolder, os.path.splitext(file_name)[0] + '_multiplot.png'),
                bbox_inches='tight')
    plt.show()
    plt.close()

    # calculate Fourier coefficients
    coordinates = sample_equidist(res, 100)
    # check the correct order of x axis - from 0 to 1 and back
    if coordinates[0][1] >= 0.99:
        coordinates = coordinates[::-1]
    # close the outline
    new_dat = np.array([np.zeros(coordinates.shape[0]), coordinates[::-1, 1]]).T
    new_dat = new_dat[1:(coordinates.shape[0] - 2), :]  # remove first and last points (duplicities)
    outline = np.concatenate((coordinates, new_dat), axis=0)

    coefs = ef_coefs(outline, n_harmonics=7)
    # read the scaler and the model for a prediction
    scaler = joblib.load('utils/scaler.save')
    classifier = joblib.load('utils/the_model.save')

    # predict with new data
    X = scaler.transform(coefs.reshape(1, -1))

    y_pred = classifier.predict(X)
    if y_pred == 1:
        y_pred = "Female"
    else:
        y_pred = "Male"
    y_prob_M = classifier.predict_proba(X)[:, 0]  # prob Male
    y_prob_F = classifier.predict_proba(X)[:, 1]  # prob Female

    print(f'Predicted sex: {y_pred}')
    print(f'Probability male: {y_prob_M}')
    print(f'Probability female: {y_prob_F}')
    print(f'Processing image {file_name} finished.')
    print(f'---------------------------------------')

    # save coefficients and predictions to the file
    coefs = pd.DataFrame(coefs)
    coefs.loc[len(coefs)] = [file_name]
    coefs.loc[len(coefs)] = y_pred
    coefs.loc[len(coefs)] = y_prob_M
    coefs.loc[len(coefs)] = y_prob_F

    data_coefs.append(pd.DataFrame(coefs))

sample_df = pd.concat(data_coefs, axis=1)
sample_df = sample_df.transpose()
sample_df.to_csv(os.path.join(folder, 'ef_coefs_' + datetime.datetime.today().strftime('%Y-%m-%d') + '.csv'),
                 index=False)
