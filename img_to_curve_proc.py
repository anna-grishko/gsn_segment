import os

import cv2
import numpy as np
import seaborn as sns
import shapefile
from matplotlib import pyplot as plt

from img_preproc_utils import ImgProcessor, ContourGSN, transform_gsn

sns.set_theme(style="ticks")  # , font_scale=1.25
sns.set_context("talk")
# the assumption is a clean img with readable GSN curve
folder = '../data/input_data'
collection = 'for_plots'

imgs = os.listdir(os.path.join(folder, collection))

for file_name in imgs:

    img = ImgProcessor.read(os.path.join(folder, collection), file_name)
    # file_name = str.split(file_name, '_')[0] + '_' + str.split(file_name, '_')[1]  # for Simon
    print('Processing image' + file_name)
    # print("- Number of Pixels: " + str(img.size))
    # print("- Shape/Dimensions: " + str(img.shape))

    img_new = img.resize()
    img_gray, img_blur, img_open, gsn_mask = img_new.mask()

    img_contour = ContourGSN.find_contour(img_new, gsn_mask)
    img_contour.show_contours()
    img_contour.segment_gsn()

    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax.imshow(cv2.cvtColor(img_contour.img, cv2.COLOR_BGR2RGB), cmap=plt.cm.Spectral)
    # ax.plot(img_contour.gsn_contour_flip[:, 1], img_contour.gsn_contour_flip[:, 0], '-r', linewidth=2)
    # ax.plot(img_contour.gsn_curve_flip[:, 1], img_contour.gsn_curve_flip[:, 0], '-b', linewidth=2)
    # plt.show()

    res = transform_gsn(img_contour.points_flip[0], img_contour.points_flip[1], img_contour.gsn_curve_flip)

    # Save results: each img into separate folder with img id name
    # inside: cropped resized img, gsn_contour before transform, gsn contour after transform
    # new_folder = os.path.join(os.path.split(folder)[0], 'Simon-redo')
    new_folder = os.path.join(os.path.split(folder)[0], 'segmented_curves/final', collection)
    # new_folder = folder
    new_subfolder = os.path.splitext(file_name)[0]
    if not os.path.exists(os.path.join(new_folder, new_subfolder)):
        os.makedirs(os.path.join(new_folder, new_subfolder))
    cv2.imwrite(os.path.join(new_folder, new_subfolder, os.path.splitext(file_name)[0] + '_gsn_img.png'),
                img_contour.img_flip)
    np.savetxt(os.path.join(new_folder, new_subfolder, 'gsn_contour.csv'), img_contour.gsn_contour_flip, delimiter=",")
    np.savetxt(os.path.join(new_folder, new_subfolder, 'gsn_curve.csv'), img_contour.gsn_curve_flip, delimiter=",")
    np.savetxt(os.path.join(new_folder, new_subfolder, 'gsn_curve_registered.csv'), res, delimiter=",")

    # Save all csvs to one folder
    np.savetxt(os.path.join(os.path.split(folder)[0], 'segmented_curves', 'gsn_curve_all', 'gsn_curve_' +
                            collection + '_' + os.path.splitext(file_name)[0] + '.csv'),
               img_contour.gsn_curve_flip, delimiter=",")
    np.savetxt(os.path.join(os.path.split(folder)[0], 'segmented_curves', 'gsn_curve_registered', 'gsn_curve_reg_' +
                            collection + '_' + os.path.splitext(file_name)[0] + '.csv'),
               res, delimiter=",")

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
    # ax[0].imshow(cv2.cvtColor(img_new.img, cv2.COLOR_BGR2RGB), cmap=plt.cm.Spectral)
    # ax[1].imshow(img_blur, cmap='gray')
    # ax[2].imshow(gsn_mask)
    # ax[0].imshow(cv2.cvtColor(img_new.img, cv2.COLOR_BGR2RGB), cmap=plt.cm.Spectral)
    # for contour in img_contour.contours:
    #     ax[0].plot(contour[:, 1], contour[:, 0], linewidth=3, markersize=12, zorder=1)
    # ax[0].scatter(img_contour.points[0][1], img_contour.points[0][0], s=120, marker='X', color='red', zorder=2)
    # ax[0].scatter(img_contour.points[1][1], img_contour.points[1][0], s=120, marker='X', color='red', zorder=3)
    ax.plot(res[:, 1], res[:, 0], '-r', linewidth=3)
    ax.set_box_aspect(gsn_mask.shape[0] / gsn_mask.shape[1])
    ax.set_ylim(ymin=0)
    ax.axis('equal')
    plt.savefig(os.path.join(new_folder, new_subfolder, os.path.splitext(file_name)[0] + '_multiplot2.png'),
                bbox_inches='tight')
    plt.savefig(os.path.join(new_folder, new_subfolder, os.path.splitext(file_name)[0] + '_multiplot2.eps'),
                format="eps", dpi=600, bbox_inches="tight", transparent=True)
    # plt.show()
    plt.close()

    # # Simplified plot
    # fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    # ax[0].imshow(cv2.cvtColor(img_new.img, cv2.COLOR_BGR2RGB), cmap=plt.cm.Spectral)
    # ax[1].imshow(img_blur, cmap='gray')
    # for contour in img_contour.contours:
    #     ax[1].plot(contour[:, 1], contour[:, 0], linewidth=1.2)
    # ax[1].plot(img_contour.points[0][1], img_contour.points[0][0], 'xr')
    # ax[1].plot(img_contour.points[1][1], img_contour.points[1][0], 'xr')
    # ax[2].plot(res[:, 1], res[:, 0], '-r', linewidth=1.2)
    # ax[2].set_box_aspect(gsn_mask.shape[0] / gsn_mask.shape[1])
    # ax[2].set_ylim(ymin=0)
    # ax[2].axis('equal')
    # plt.savefig(os.path.join(new_folder, new_subfolder, os.path.splitext(file_name)[0] + '_multiplot.png'),
    #             bbox_inches='tight')
    # # plt.show()
    # plt.close()

    # # For presentation
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax.imshow(gsn_mask)
    # plt.show()
    #
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax.imshow(cv2.cvtColor(img_new.img, cv2.COLOR_BGR2RGB), cmap=plt.cm.Spectral)
    # for contour in img_contour.contours:
    #     ax.plot(contour[:, 1], contour[:, 0], linewidth=5, c='r')
    # ax.plot(img_contour.points[0][1], img_contour.points[0][0], 'xr', markersize=15, markeredgewidth=5,
    #         markeredgecolor='purple', markerfacecolor='purple')
    # ax.plot(img_contour.points[1][1], img_contour.points[1][0], 'xr', markersize=15, markeredgewidth=5,
    #         markeredgecolor='purple', markerfacecolor='purple')
    # plt.show()
    #
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax.plot(res[:, 1], res[:, 0], '-r', linewidth=3)
    # ax.set_box_aspect(gsn_mask.shape[0] / gsn_mask.shape[1])
    # ax.set_ylim(ymin=0)
    # ax.axis('equal')
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # ax.imshow(cv2.cvtColor(img_contour.img.img, cv2.COLOR_BGR2RGB), cmap=plt.cm.Spectral)
    # for contour in img_contour.contours:
    #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    # ax.axis('image')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.savefig(os.path.join(new_folder, new_subfolder, os.path.splitext(file_name)[0] + '_contours.png'),
    #             bbox_inches='tight')
    # plt.show()

    # save as a shape file - polygon
    # shape = [list(zip(res[:, 1], res[:, 0]))]
    # w = shapefile.Writer(os.path.join(new_folder, new_subfolder, 'polygon'))
    # w.poly(shape)
    # w.field('FIRST_FLD', 'C', '40')
    # w.field('SECOND_FLD', 'C', '40')
    # w.record('First', 'Polygon')
    # w.close()
