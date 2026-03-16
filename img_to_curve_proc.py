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
    # img_gray, img_blur, img_open, gsn_mask = img_new.mask()
    masked_img = img_new.mask()

    img_contour = ContourGSN.find_contour(img_new, masked_img['mask'])
    img_contour.show_contours()
    img_contour.segment_gsn()

    res = transform_gsn(img_contour.flipped_data['points'][0], img_contour.flipped_data['points'][1], img_contour.flipped_data['curve'])

    # Save results: each img into separate folder with img id name
    # inside: cropped resized img, gsn_contour before transform, gsn contour after transform
    new_folder = os.path.join(os.path.split(folder)[0], 'segmented_curves/final', collection)
    new_subfolder = os.path.splitext(file_name)[0]
    if not os.path.exists(os.path.join(new_folder, new_subfolder)):
        os.makedirs(os.path.join(new_folder, new_subfolder))
    cv2.imwrite(os.path.join(new_folder, new_subfolder, os.path.splitext(file_name)[0] + '_gsn_img.png'),
                img_contour.flipped_data['img'])
    np.savetxt(os.path.join(new_folder, new_subfolder, 'gsn_contour.csv'), img_contour.flipped_data['contour'], delimiter=",")
    np.savetxt(os.path.join(new_folder, new_subfolder, 'gsn_curve.csv'), img_contour.flipped_data['curve'], delimiter=",")
    np.savetxt(os.path.join(new_folder, new_subfolder, 'gsn_curve_registered.csv'), res, delimiter=",")


    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax[0].imshow(cv2.cvtColor(img_new.img, cv2.COLOR_BGR2RGB), cmap=plt.cm.Spectral)
    # ax[1].imshow(img_blur, cmap='gray')
    # ax[2].imshow(gsn_mask)
    ax.imshow(cv2.cvtColor(img_new.img, cv2.COLOR_BGR2RGB), cmap=plt.cm.Spectral)
    for contour in img_contour.contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=3, markersize=12, zorder=1)
    ax.scatter(img_contour.points[0][1], img_contour.points[0][0], s=120, marker='X', color='red', zorder=2)
    ax.scatter(img_contour.points[1][1], img_contour.points[1][0], s=120, marker='X', color='red', zorder=3)
    ax.annotate("A", (img_contour.points[0][1], img_contour.points[0][0]), xytext=(10, 10), textcoords='offset points', color='red', fontsize=50)
    ax.annotate("B", (img_contour.points[1][1], img_contour.points[1][0]), xytext=(10, 10), textcoords='offset points', color='red', fontsize=50)
    # ax[1].plot(res[:, 1], res[:, 0], '-r', linewidth=3)
    # ax[1].set_box_aspect(gsn_mask.shape[0] / gsn_mask.shape[1])
    # ax[1].set_ylim(ymin=0)
    # ax[1].axis('equal')
    plt.savefig(os.path.join(new_folder, new_subfolder, os.path.splitext(file_name)[0] + '_multiplot.png'),
                bbox_inches='tight')
    plt.savefig(os.path.join(new_folder, new_subfolder, os.path.splitext(file_name)[0] + '_multiplot.eps'),
                format="eps", dpi=100, bbox_inches="tight", transparent=True)
    # plt.show()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(res[:, 1], res[:, 0], '-r', linewidth=3)
    ax.set_box_aspect(masked_img['mask'].shape[0] / masked_img['mask'].shape[1])
    ax.set_ylim(ymin=0)
    ax.axis('equal')
    plt.savefig(os.path.join(new_folder, new_subfolder, os.path.splitext(file_name)[0] + '_multiplot2.png'),
                bbox_inches='tight')
    plt.savefig(os.path.join(new_folder, new_subfolder, os.path.splitext(file_name)[0] + '_multiplot2.eps'),
                format="eps", dpi=600, bbox_inches="tight", transparent=True)
    # plt.show()
    plt.close()
