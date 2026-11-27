import os
import cv2
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from utils.img_preproc_utils import ImgProcessor, ContourGSN, transform_gsn

sns.set_theme(style="ticks")  # , font_scale=1.25
sns.set_context("talk")
# the assumption is a clean img with readable GSN curve
folder = 'data/input_data'
collection = 'for_plots'

imgs = os.listdir(os.path.join(folder, collection))

for file_name in imgs:

    img = ImgProcessor.read(os.path.join(folder, collection), file_name)
    print('Processing image' + file_name)
    # print("- Number of Pixels: " + str(img.size))
    # print("- Shape/Dimensions: " + str(img.shape))

    img_new = img.resize()
    masked_img = img_new.mask()

    img_contour = ContourGSN.find_contour(img_new, masked_img['mask'])
    # img_contour.show_contours()
    img_contour.segment_gsn()

    res = transform_gsn(img_contour.flipped_data['points'][0], img_contour.flipped_data['points'][1], img_contour.flipped_data['curve'])

    # Save results: each img into separate folder with img id name
    new_folder = os.path.join(os.path.split(folder)[0], 'segmented_curves/final', collection)
    new_subfolder = os.path.splitext(file_name)[0]
    if not os.path.exists(os.path.join(new_folder, new_subfolder)):
        os.makedirs(os.path.join(new_folder, new_subfolder))
    np.savetxt(os.path.join(new_folder, new_subfolder, 'gsn_contour.csv'), img_contour.flipped_data['contour'], delimiter=",")
    np.savetxt(os.path.join(new_folder, new_subfolder, 'gsn_curve.csv'), img_contour.flipped_data['curve'], delimiter=",")
    np.savetxt(os.path.join(new_folder, new_subfolder, 'gsn_curve_registered.csv'), res, delimiter=",")
