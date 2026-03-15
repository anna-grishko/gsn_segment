import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import morphology, filters, measure, img_as_ubyte
from skimage.feature import canny


# Image subprocessor class
class ImgProcessor(object):
    def __init__(self, img):
        self.img = img
        self.size = img.size
        self.shape = img.shape

    @classmethod
    def read(cls, folder, file_name):
        img = cv2.imread(os.path.join(folder, file_name))
        return cls(img)

    def resize(self, height=500):
        # Select ROI
        r = cv2.selectROI("select the area and press enter", self.img)
        # close the window
        cv2.destroyAllWindows()
        # Crop image
        cropped_image = self.img[int(r[1]):int(r[1] + r[3]),
                        int(r[0]):int(r[0] + r[2])]
        # resize
        scale = height / cropped_image.shape[0]
        width = int(cropped_image.shape[1] * scale)
        new_size = (width, height)
        img_new = cv2.resize(cropped_image, new_size, interpolation=cv2.INTER_CUBIC)
        return ImgProcessor(img_new)

    def mask(self, blur=cv2.bilateralFilter):
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        if blur == False:
            img_blur = img_gray
        else:
            img_blur = blur(img_gray, 15, 100, 5)
        # TODO pass various arguments as *args (?)
        # img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        # img_blur = cv2.medianBlur(img_gray, 3)

        # Simple Otsu threshold
        # TODO: add other methods, e.g. closing
        img_open = morphology.area_opening(img_blur)
        image_th_open = filters.threshold_otsu(img_open)
        mask = np.where(img_open <= image_th_open, 1, 0)

        return [img_gray, img_blur, img_open, mask]


class ContourGSN:
    def __init__(self, img, mask, contours, gsn_contour_idx, gsn_contour, gsn_curve, gsn_contour_flip, gsn_curve_flip,
                 points, points_flip, img_flip):
        self.img = img
        self.size = img.size
        self.shape = img.shape
        self.mask = mask
        self.contours = contours
        self.gsn_contour_idx = gsn_contour_idx
        self.gsn_contour = gsn_contour
        self.gsn_curve = gsn_curve
        self.gsn_curve_flip = gsn_curve_flip
        self.gsn_contour_flip = gsn_contour_flip
        self.points = points
        self.points_flip = points_flip
        self.img_flip = img_flip

    @classmethod
    def find_contour(cls, img, mask, shade=False):
        if shade:
            hsv_img = cv2.cvtColor(img.img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_img)
            s = filters.gaussian(s, sigma=3)
            edges = canny(s, sigma=3)
            contours, hierarchy = cv2.findContours(img_as_ubyte(edges), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_new = []
            for contour in contours:
                contours_new.append(contour[:, 0, :])
            contours = contours_new
        else:
            contours = measure.find_contours(mask, level=0.5)
        # GSN is never a closed contour
        # -> find the longer unclosed contour
        # for now substitute by the longest contour
        contour_idx = np.argmax([contour.shape[0] for contour in contours])  # find index of the biggest contour
        gsn_contour = contours[contour_idx]

        return ContourGSN(img, mask, contours, contour_idx, gsn_contour, [], [], [], [], [], [])

    def show_contours(self, all=True):
        fig, ax = plt.subplots()
        # img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        ax.imshow(cv2.cvtColor(self.img.img, cv2.COLOR_BGR2RGB), cmap=plt.cm.Spectral)
        # ax.imshow(self.img) #, cmap=plt.cm.gray)
        if all:
            for contour in self.contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        else:
            ax.plot(self.gsn_contour[:, 1], self.gsn_contour[:, 0], linewidth=2)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    def segment_gsn(self):
        point_1, point_2 = get_landmarks(self.img.img)
        # find the closest point to the first point
        distances = np.linalg.norm(self.gsn_contour - point_1, axis=1)
        index_1 = np.argmin(distances)
        # find the closest point to the second point
        distances = np.linalg.norm(self.gsn_contour - point_2, axis=1)
        index_2 = np.argmin(distances)

        point_piriform = self.gsn_contour[index_1].copy()
        point_ischial = self.gsn_contour[index_2].copy()

        # the final curve segment contains points between those indexes
        if index_1 < index_2:
            final_curve = self.gsn_contour[index_1:index_2]
        else:
            final_curve = self.gsn_contour[index_2:index_1]

        ### step 1 completed: curve segmented
        # origin [0,0]
        # detect if the acr is facing up or down and automatically flip
        # point 1 = piriformis tubercle, point 2 = tip of the ischial spine
        rows, cols, channels = self.shape
        n_points = len(final_curve)
        gsn_curve_flip = final_curve.copy()
        gsn_contour_flip = self.gsn_contour.copy()
        img_flip = self.img.img.copy()
        point_piriform_flip = point_piriform.copy()
        point_ischial_flip = point_ischial.copy()
        # IF point piriformis tubercle on the left - OK
        # IF point 1 on the right

        if point_piriform_flip[1] > point_ischial_flip[1]:
            # 1. flip around y-axis: cv2.flip(img_blur1, 1)
            img_flip = cv2.flip(img_flip, 1)
            gsn_curve_flip[:, 1] = cols - 1 - gsn_curve_flip[:, 1]
            gsn_contour_flip[:, 1] = cols - 1 - gsn_contour_flip[:, 1]
            point_piriform_flip[1] = cols - 1 - point_piriform_flip[1]
            point_ischial_flip[1] = cols - 1 - point_ischial_flip[1]

        # IF curve facing upwards <=> curve midpoint y coord < curve y endpoints
        if gsn_curve_flip[n_points // 2][0] < point_piriform_flip[0]:
            img_flip = cv2.flip(img_flip, 0)
            gsn_curve_flip[:, 0] = rows - 1 - gsn_curve_flip[:, 0]
            gsn_contour_flip[:, 0] = rows - 1 - gsn_contour_flip[:, 0]
            # recalculate landmarks location
            point_piriform_flip[0] = rows - 1 - point_piriform_flip[0]
            point_ischial_flip[0] = rows - 1 - point_ischial_flip[0]
        # IF curve facing downwards - OK

        self.img_flip = img_flip
        self.gsn_curve = final_curve
        self.gsn_curve_flip = gsn_curve_flip
        self.gsn_contour_flip = gsn_contour_flip
        self.points = [point_piriform, point_ischial]
        self.points_flip = [point_piriform_flip, point_ischial_flip]


def transform_gsn(point_piriform, point_ischial, gsn_curve):
    ### Affine Transformation (translation and rotation)
    # origin (pocatek souradnic) [row,column]=[y,x]
    point_origin = [0, 0]
    # calculate translation according to the point_1
    shift = point_origin - point_piriform

    M_shift = np.float32([[1, 0, shift[1]], [0, 1, shift[0]], [0, 0, 1]])
    point_ischial_shift = point_ischial + shift

    theta = np.arctan((point_ischial_shift[0]) / (point_ischial_shift[1]))
    M = cv2.getRotationMatrix2D((point_origin), theta * 180 / np.pi, 1)

    # translate points
    shifted_points = gsn_curve + shift
    # rotate points
    M_new = np.array([[M[0][0], -M[0][1]],
                      [M[0][1], M[0][0]]])
    transformed_points = M_new.dot(shifted_points.T).T
    point_ischial = M_new.dot(point_ischial_shift.T).T

    # Scale
    transformed_points = transformed_points / point_ischial[1]

    return transformed_points


class CoordinateStore:
    def __init__(self, img):
        self.points = []
        self.img = img

    def click_event(self, event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)
            # img = self.img.copy()

            # displaying selected points on the image window
            cv2.circle(self.img, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
            self.points.append((x, y))
            cv2.imshow('Select two landmarks', self.img)


def get_landmarks(image):
    ### restrict contour
    image_temp = image.copy()
    coordinate_store = CoordinateStore(image_temp)
    # displaying the image
    # gsn_contour = np.array([np.array([x, y]) for y, x in self.gsn_contour.astype(int)])
    # ctr = np.array(self.gsn_contour.astype(int)).reshape((-1, 1, 2)).astype(np.int32)
    # plot_img = cv2.drawContours(self.img.img, [gsn_contour], 0, (0, 0, 255), 2)
    cv2.imshow('Select two landmarks', image_temp)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('Select two landmarks', coordinate_store.click_event)
    # wait for a key to be pressed to exit
    cv2.waitKey(0)
    # close the window
    cv2.destroyAllWindows()

    # if coordinate_store1.points[0][0] < coordinate_store1.points[1][0]:
    point_1 = [coordinate_store.points[0][1], coordinate_store.points[0][0]]
    point_2 = [coordinate_store.points[1][1], coordinate_store.points[1][0]]

    return [point_1, point_2]
